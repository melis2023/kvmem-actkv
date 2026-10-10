"""Real-model session spill/restore, LRU and failure regression; downloads nothing."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mtp', action='store_true')
p.add_argument('--lines', type=int, default=280)
p.add_argument('--gpu-layers', default='99', help='layers to offload; use all for multi-GPU')
p.add_argument('--mtp-state', choices=('auto', 'snapshots', 'replay'), default='replay',
               help='auto supports model shapes outside the specialized 27B replay kernel')
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['KVMEM_TRACE'] = '1'
checks = []

def check(name, condition):
    checks.append({'name': name, 'pass': bool(condition)})
    print(('PASS ' if condition else 'FAIL ') + name, flush=True)
    if not condition:
        raise AssertionError(name)

def request(base, path, data=None):
    req = urllib.request.Request(base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

codes = {'A': 'ALPHA-731', 'B': 'BETA-118', 'C': 'GAMMA-905', 'D': 'DELTA-426'}

def notes(name):
    lines = [f'Channel {name} note {i:03d}: routing entry {i} is stable and requires no action.'
             for i in range(a.lines)]
    return '\n'.join(lines + [f'The {name} code is {codes[name]}.',
        f'What is the {name} code? Answer with the code only.'])

def run(label, ram=8, disk=8, sequence=('A', 'B', 'C', 'A', 'B', 'C'), corrupt=False,
        missing=False, lock_restore=False):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    cache = out / (label + '-cache-' + uuid.uuid4().hex[:8]); cache.mkdir()
    logpath = out / (label + '.log')
    args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
        '-c', '16384', '-n', '32', '-ngl', str(a.gpu_layers), '--reasoning-effort', 'none', '--temp', '0',
        '--no-webui', '--kv-dtype', 'q8_0', '-fa', 'on', '--kvmem-budget', '2048',
        '--kvmem-gen-reserve', '512', '--spec-type', 'draft-mtp' if a.mtp else 'none',
        '--kvmem-mtp-state', a.mtp_state,
        '--kvmem-conversations', '3', '--kvmem-session-ram-gb', str(ram),
        '--kvmem-session-nvme-gb', str(disk), '--kvmem-session-cache-dir', str(cache)]
    histories = {k: [{'role': 'system', 'content': 'Read the notes and answer with the code only.'}] for k in codes}
    turns = []
    with logpath.open('wb') as log:
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=log,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            end = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited: {logpath}')
                try:
                    if request(base, '/health')[0] == 200: break
                except OSError: pass
                if time.monotonic() > end: raise TimeoutError('server startup')
                time.sleep(.5)
            for index, name in enumerate(sequence):
                locked_handle = None
                if (corrupt or missing or lock_restore) and index == 3:
                    # A is the oldest resident on disk after A/B/C. Derive the
                    # exact file from the diagnostic; never touch other runs.
                    text = logpath.read_text(encoding='utf-8', errors='replace')
                    ids = re.findall(r'session_select id=(\d+)', text)
                    candidates = sorted(cache.glob(f'run-*/{ids[0]}-*.kv')) if ids else []
                    check(label + ' has an A snapshot for fault injection', len(candidates) > 0)
                    victim = candidates[len(candidates)//2]
                    if missing:
                        victim.unlink()
                    elif lock_restore:
                        import ctypes
                        from ctypes import wintypes
                        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
                        kernel.CreateFileW.restype = wintypes.HANDLE
                        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                        kernel.CloseHandle.restype = wintypes.BOOL
                        locked_handle = kernel.CreateFileW(str(victim), 0x80000000, 0, None, 3, 0x80, None)
                        if locked_handle == wintypes.HANDLE(-1).value:
                            raise ctypes.WinError(ctypes.get_last_error())
                    else:
                        with victim.open('r+b') as f:
                            f.seek(-1, 2); byte = f.read(1); f.seek(-1, 2); f.write(bytes([byte[0] ^ 1]))
                history = histories[name]
                history.append({'role': 'user', 'content': notes(name) if len(history) == 1 else
                    f'What is the {name} code from the notes? Answer with the code only.'})
                body = {'messages': history,
                    'max_tokens': 32, 'temperature': 0, 'reasoning_effort': 'none',
                    'kvmem': {'conversation_id': name}}
                if locked_handle is not None:
                    before = request(base, '/slots')[1][0]['kvmem']['conversations']
                    try:
                        status, result = request(base, '/v1/chat/completions', body)
                        check(label + ' locked restore reports HTTP 503', status == 503)
                        failed = request(base, '/slots')[1][0]['kvmem']['conversations']
                        check(label + ' failed switch keeps all sessions', failed['count'] == before['count'] and
                            failed['evictions'] == before['evictions'])
                        check(label + ' failed switch exposes no partial active KV', failed['active'] == -1)
                    finally:
                        kernel.CloseHandle(locked_handle)
                    # Resume the OLD active session first, then retry A. This
                    # traverses mixed RAM/disk ownership in both directions.
                    old = histories['C']
                    old.append({'role': 'user', 'content': 'Repeat the C code, nothing else.'})
                    status, recovered = request(base, '/v1/chat/completions', {
                        **body, 'messages': old, 'kvmem': {'conversation_id': 'C'}})
                    check(label + ' old C can resume after failed switch', status == 200)
                    answer = recovered['choices'][0]['message']['content'] or ''
                    check(label + ' recovered C keeps its own KV', codes['C'] in answer and codes['A'] not in answer)
                    check(label + ' recovered C retains cached prefix', recovered['usage']['prompt_cache_hit_tokens'] > 1024)
                    old.append({'role': 'assistant', 'content': answer})
                status, result = request(base, '/v1/chat/completions', body)
                check(f'{label} {index}:{name} HTTP 200 ({result.get("error", "")})', status == 200)
                answer = result['choices'][0]['message']['content'] or ''
                history.append({'role': 'assistant', 'content': answer})
                check(f'{label} {index}:{name} isolated answer', codes[name] in answer and
                    not any(code in answer for key, code in codes.items() if key != name))
                counters = request(base, '/slots')[1][0]['kvmem']['conversations']
                check(f'{label} {index} disk quota and session count',
                    counters['disk_bytes'] <= counters['disk_bytes_max'] and counters['count'] <= 3)
                on_disk = sum(path.stat().st_size for path in cache.glob('run-*/*')
                              if path.suffix in ('.kv', '.tmp'))
                check(f'{label} {index} charged disk bytes match files', counters['disk_bytes'] == on_disk)
                turns.append({'channel': name, 'answer': answer, 'usage': result['usage'], 'counters': counters})
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            # These servers are forcibly stopped on Windows, so their C++
            # destructors cannot clean up. Remove only numeric snapshot files
            # inside this invocation's newly created private test directory.
            cache_root = cache.resolve()
            for path in cache.glob('run-*/*'):
                if (re.fullmatch(r'\d+-\d+', path.stem) and path.suffix in ('.kv', '.tmp') and
                        not path.is_symlink() and path.resolve().is_relative_to(cache_root)):
                    path.unlink()
            for path in cache.glob('run-*'):
                if not path.is_symlink() and path.resolve().is_relative_to(cache_root):
                    try: path.rmdir()
                    except OSError: pass  # leave any unrecognized files alone
            try: cache.rmdir()
            except OSError: pass
    text = logpath.read_text(encoding='utf-8', errors='replace')
    if a.mtp:
        check(label + ' MTP follower was exercised', 'spec_start type=draft-mtp' in text and 'mtp_follow' in text)
    (out / (label + '.json')).write_text(json.dumps(turns, indent=2), encoding='utf-8')
    return turns, text

try:
    reference, trace = run('ram-reference')
    reservations = [int(n) for n in re.findall(r'session_select .*?reserve=(\d+)', trace)]
    check('reference produces reservations', len(reservations) == 6)
    measured, trace = run('measured-ram', ram=reference[0]['counters']['bytes'] * 2.5 / 2**30,
        sequence=('A','B','A'))
    check('measured RAM that fits is kept despite pessimistic forecast', measured[-1]['counters']['spills'] == 0)
    check('measured RAM retains A prefix', measured[-1]['usage']['prompt_cache_hit_tokens'] > 1024)
    # Set the soft cap BELOW one measured active session, including below the
    # old minimum checkpoint reservation. Startup and every request must work.
    ram = reference[0]['counters']['bytes'] * 0.75 / 2**30
    cached, trace = run('disk-roundtrip', ram=ram)
    check('disk actually spills and restores', 'session_spill' in trace and 'session_restore' in trace)
    check('active session can exceed RAM soft cap', all(t['counters']['bytes'] > t['counters']['bytes_max'] for t in cached))
    check('soft overage warns without clearing KV', 'session RAM soft limit exceeded' in trace and 'clearing completed cache' not in trace)
    max_snapshot = max(int(n) for n in re.findall(r'session_spill .*?disk_bytes=(\d+)', trace))
    for i in range(3, 6):
        check(f'disk turn {i} reuses its cached prefix', cached[i]['usage']['prompt_cache_hit_tokens'] > 1024)
        check(f'disk turn {i} matches RAM answer', cached[i]['answer'] == reference[i]['answer'])
    count_lru, trace = run('count-lru', ram=ram, sequence=('A','B','C','A','D','C','B'))
    check('disk-mode count cap uses LRU', 'reason=lru' in trace and count_lru[4]['counters']['evictions'] > 0)
    check('count cap preserves recently used C', count_lru[5]['usage']['prompt_cache_hit_tokens'] > 1024)
    check('count cap discards oldest B', count_lru[6]['usage']['prompt_cache_hit_tokens'] == 0)
    broken, trace = run('corrupt-fallback', ram=ram, sequence=('A','B','C','A'), corrupt=True)
    check('corrupt snapshot counted', broken[-1]['counters']['disk_errors'] >= 1)
    check('corrupt snapshot recomputed', broken[-1]['usage']['prompt_cache_hit_tokens'] == 0)
    absent, trace = run('missing-fallback', ram=ram, sequence=('A','B','C','A'), missing=True)
    check('missing snapshot recomputed', absent[-1]['usage']['prompt_cache_hit_tokens'] == 0)
    if os.name == 'nt':
        retried, trace = run('locked-restore', ram=ram, sequence=('A','B','C','A'), lock_restore=True)
        check('retry A retains its cached prefix', retried[-1]['usage']['prompt_cache_hit_tokens'] > 1024)
    lru, trace = run('disk-lru', ram=ram, disk=max_snapshot * 1.05 / 2**30,
        sequence=('A','B','C','B','A'))
    check('disk LRU removed the oldest snapshot', 'reason=disk_lru' in trace)
    check('recent B survives disk LRU', lru[3]['usage']['prompt_cache_hit_tokens'] > 1024)
    check('old A was evicted by disk LRU', lru[4]['usage']['prompt_cache_hit_tokens'] == 0)
    # Capacity below any meaningful session file forces LRU discard; next A is
    # a miss, while inference and quota accounting remain healthy.
    evicted, trace = run('tiny-disk', ram=ram, disk=0.00001, sequence=('A','B','C','A'))
    check('full disk causes LRU eviction', evicted[-1]['counters']['evictions'] > 0)
    check('evicted A recomputes', evicted[-1]['usage']['prompt_cache_hit_tokens'] == 0)
    unlimited, trace = run('unlimited-ram', ram=0, sequence=('A','B','A'))
    check('zero RAM cap remains unlimited in disk mode', unlimited[-1]['counters']['bytes_max'] == 0)
    check('unlimited RAM avoids unnecessary spills', unlimited[-1]['counters']['spills'] == 0)
    check('unlimited RAM reuses A', unlimited[-1]['usage']['prompt_cache_hit_tokens'] > 1024)
finally:
    (out / 'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
