#!/usr/bin/env python3
"""Run the complete repository Task 1 with GDN output fusion off/on in ABBA order."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, default=root / 'build-win/bin/llama-kvmem-server.exe')
    parser.add_argument('--model', type=Path, default=root.parent / 'models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')
    parser.add_argument('--mmproj', type=Path, default=root.parent / 'models/mmproj-Q8_0.gguf')
    parser.add_argument('--output', type=Path, default=root / 'artifacts/gdn-task1')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='reuse fully completed groups with matching options')
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 1:
        parser.error('--repeats and --rounds must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'server': str(args.server.resolve()), 'model': str(args.model.resolve()), 'mmproj': str(args.mmproj.resolve()),
        'server_sha256': hashlib.sha256(args.server.read_bytes()).hexdigest(),
        'patch_sha256': hashlib.sha256((root / 'patches/gdn-output-fusion.patch').read_bytes()).hexdigest(),
        'canary_sha256': hashlib.sha256((root / 'scripts/multimodal_canary.py').read_bytes()).hexdigest(),
        'repeats': args.repeats, 'rounds': args.rounds, 'warmup_runs': 1,
        'task': '12000 words -> 896x896 tricolor image -> HTML/SVG code; thinking budget 128, max output 512',
        'gpu': 'physical NVML index 1, CUDA_VISIBLE_DEVICES=1 with PCI_BUS_ID order',
        'trace': False,
    }
    config_path = args.output / 'config.json'
    if args.resume:
        if not config_path.exists():
            parser.error('--resume requires the original config.json')
        original = json.loads(config_path.read_text(encoding='utf-8'))
        mismatches = [key for key, value in config.items() if key != 'created' and original.get(key) != value]
        if mismatches:
            parser.error('--resume configuration changed: ' + ', '.join(mismatches))
    else:
        save(config_path, config)
    rows, groups = [], []
    for round_id in range(args.rounds):
        modes = ('baseline', 'fused') if round_id % 2 == 0 else ('fused', 'baseline')
        for mode in modes:
            folder = args.output / f'round-{round_id}-{mode}'
            folder.mkdir(exist_ok=True)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = [sys.executable, str(root / 'scripts/multimodal_canary.py'),
                '--binary', str(args.server.resolve()), '--model', str(args.model.resolve()),
                '--mmproj', str(args.mmproj.resolve()), '--gpu-index', '1', '--query-replay', 'auto',
                '--query-policy', 'user', '--mtp-state', 'replay', '--mtp', '3', '--kv', 'q8_0',
                '--draft-kv', 'f16', '--image-max-tokens', '1024', '--budget', '36864', '--reserve', '16384',
                '--ctx', '262144', '--batch', '512', '--long-words', '12000', '--quick', '--performance',
                '--thinking-budget', '128', '--warmup-runs', '1', '--performance-runs', str(args.repeats),
                '--startup-timeout', '900', '--port', str(port), '--folder', str(folder)]
            env = dict(os.environ, PYTHONUTF8='1', KVMEM_GDN_OUT_FUSION='1' if mode == 'fused' else '0',
                       KVMEM_GDN_OUT_FUSION_TRACE='0')
            for name in ('KVMEM_TRACE', 'KVMEM_PERF', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION'):
                env.pop(name, None)
            completed = False
            if args.resume and (folder / 'summary.json').exists():
                old = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
                options = old.get('options', {})
                launch = json.loads((folder / 'launch.json').read_text(encoding='utf-8'))
                completed = (options.get('performance_runs') == args.repeats and options.get('ctx') == 262144 and
                             options.get('long_words') == 12000 and len(old.get('requests', [])) == 3 * (1 + args.repeats) and
                             launch.get('fusion') == env['KVMEM_GDN_OUT_FUSION'])
            print(f"{'RESUME' if completed else 'START'} round={round_id} mode={mode} folder={folder}", flush=True)
            if not completed:
                save(folder / 'launch.json', {'command': command, 'fusion': env['KVMEM_GDN_OUT_FUSION']})
                with (folder / 'console.log').open('w', encoding='utf-8') as log:
                    subprocess.run(command, env=env, cwd=root, stdout=log, stderr=log, check=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            requests = json.loads((folder / 'requests.json').read_text(encoding='utf-8'))
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            server_log = (folder / 'server.stderr.log').read_text(encoding='utf-8', errors='replace')
            device_names = re.findall(r'Device \d+: ([^,\r\n]+)', server_log)
            if device_names and device_names != ['NVIDIA GeForce RTX 5060 Ti']:
                raise RuntimeError('server did not use only the intended GPU')
            if summary.get('sampling_errors') or not summary.get('sample_count'):
                raise RuntimeError('GPU memory sampling failed')
            if not summary.get('peak_runtime_rss_mib'):
                raise RuntimeError('runtime RAM sampling failed')
            for repeat in range(args.repeats):
                prefix = f'run{repeat + 1}-' if args.repeats > 1 else ''
                stages = [x for x in requests if x['label'].startswith(prefix) and not x['label'].startswith('warmup')]
                if len(stages) != 3:
                    raise RuntimeError('incomplete Task 1')
                for item in stages:
                    response = item['response']
                    if item['status'] != 200:
                        raise RuntimeError('request failed')
                    timing = response['timings']
                    row = {'round': round_id, 'mode': mode, 'repeat': repeat,
                           'stage': item['label'][len(prefix):], 'wall_s': item['elapsed_s'],
                           'timings': timing, 'usage': response['usage'],
                           'message': response['choices'][0]['message'],
                           'request_sha256': hashlib.sha256((folder / (item['label'] + '.request.json')).read_bytes()).hexdigest()}
                    if row['stage'] == 'long-text' and timing['cache_n'] != 0:
                        raise RuntimeError('text prefill unexpectedly reused a prefix')
                    if row['stage'] == 'code-thinking' and timing['predicted_n'] != 512:
                        raise RuntimeError('code output did not reach the Task 1 token limit')
                    rows.append(row)
                    print(f"PASS {mode} run={repeat + 1} stage={row['stage']} "
                          f"prefill={timing['prompt_ms']:.2f} ms decode={timing['predicted_per_second']:.2f} tok/s "
                          f"wall={row['wall_s']:.2f} s", flush=True)
            rss = list(csv.DictReader((folder / 'rss.csv').open(encoding='utf-8')))
            vram = list(csv.DictReader((folder / 'vram.csv').open(encoding='utf-8')))
            measured = lambda x: not x['phase'].startswith('warmup') and x['phase'] != 'loading'
            group = {'round': round_id, 'mode': mode, 'peak_vram_mib': summary['peak_vram_mib'],
                     'measured_peak_vram_mib': max(float(x['used_mib']) for x in vram if measured(x)),
                     'peak_runtime_rss_mib': summary['peak_runtime_rss_mib'],
                     'measured_peak_rss_mib': max(float(x['rss_mib']) for x in rss if measured(x)),
                     'buffers': re.findall(r'(?:sched_reserve:|load_tensors:)\s+(.+?buffer size\s*=\s*[^\r\n]+)', server_log)}
            groups.append(group)
            save(args.output / 'rows.json', rows)
            save(args.output / 'groups.json', groups)
    metrics = []
    output_checks = []
    for stage in ('long-text', 'image', 'code-thinking'):
        cases = {mode: [r for r in rows if r['stage'] == stage and r['mode'] == mode] for mode in ('baseline', 'fused')}
        reference = cases['baseline'][0]
        identical = all(r['message'] == reference['message'] and r['usage'] == reference['usage'] and
                        r['request_sha256'] == reference['request_sha256'] for case in cases.values() for r in case)
        output_checks.append({'stage': stage, 'all_outputs_usage_and_requests_identical': identical})
        values = {mode: {'prefill_ms': statistics.median(r['timings']['prompt_ms'] for r in case),
                         'decode_tps': statistics.median(r['timings']['predicted_per_second'] for r in case),
                         'wall_s': statistics.median(r['wall_s'] for r in case),
                         'prompt_tokens': sorted({r['usage']['prompt_tokens'] for r in case}),
                         'generated_tokens': sorted({r['timings']['predicted_n'] for r in case})}
                  for mode, case in cases.items()}
        metrics.append({'stage': stage, **values,
                        'prefill_gain': values['baseline']['prefill_ms'] / values['fused']['prefill_ms'] - 1,
                        'decode_gain': values['fused']['decode_tps'] / values['baseline']['decode_tps'] - 1})
    whole = {}
    for mode in ('baseline', 'fused'):
        whole[mode] = statistics.median(sum(r['wall_s'] for r in rows if r['mode'] == mode and
                                            r['round'] == i and r['repeat'] == j)
                                       for i in range(args.rounds) for j in range(args.repeats))
    save(args.output / 'summary.json', {'metrics': metrics, 'task_wall_s': whole,
         'wall_time_reduction': 1 - whole['fused'] / whole['baseline'], 'output_checks': output_checks,
         'memory': groups, 'samples_per_mode': args.rounds * args.repeats})
    print(json.dumps({'metrics': metrics, 'task_wall_s': whole, 'output_checks': output_checks}, ensure_ascii=False), flush=True)
    if not all(check['all_outputs_usage_and_requests_identical'] for check in output_checks):
        raise RuntimeError('Task 1 output, usage or request differed between repeats/modes; see summary.json')


if __name__ == '__main__':
    main()
