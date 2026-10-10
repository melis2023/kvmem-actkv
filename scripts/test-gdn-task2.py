#!/usr/bin/env python3
"""Compare GDN fusion OFF/ON with one complete 256K IQ3 Task 2 per mode."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(folder):
    summary = load(folder / 'summary.json')
    requests = load(folder / 'requests.json')
    context = load(folder / 'long-context.json')
    if not context['completed'] or len(context['rounds']) != 32 or len(requests) != 33:
        raise RuntimeError(f'incomplete Task 2: {folder}')
    if summary.get('sampling_errors') or not summary.get('sample_count') or not summary.get('peak_runtime_rss_mib'):
        raise RuntimeError(f'memory sampling failed: {folder}')
    rows = []
    previous = 0
    for request in requests:
        label = request['label']
        wire = (folder / (label + '.response.txt')).read_text(encoding='utf-8')
        chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
        if request['status'] != 200 or 'data: [DONE]' not in wire or any('error' in c for c in chunks):
            raise RuntimeError(f'incomplete SSE: {folder}/{label}')
        timing = next(c['timings'] for c in reversed(chunks) if c.get('usage') and c.get('timings'))
        usage = request['usage']
        content = ''.join(choice.get('delta', {}).get('content') or '' for c in chunks for choice in c.get('choices', []))
        reasoning = ''.join(choice.get('delta', {}).get('reasoning_content') or '' for c in chunks for choice in c.get('choices', []))
        finish = next(choice['finish_reason'] for c in reversed(chunks) for choice in c.get('choices', []) if choice.get('finish_reason'))
        row = {'label': label, 'prompt_tokens': usage['prompt_tokens'],
               'new_prompt_positions': usage['prompt_tokens'] - previous,
               'usage': usage, 'timings': timing, 'wall_s': request['elapsed_s'], 'ttft_s': request['ttft_s'],
               'content': content, 'reasoning_content': reasoning, 'finish_reason': finish,
               'request_sha256': digest(folder / (label + '.request.json'))}
        if rows and usage['prompt_cache_hit_tokens'] <= 0:
            raise RuntimeError(f'lost prefix cache: {folder}/{label}')
        if timing['predicted_n'] != usage['completion_tokens']:
            raise RuntimeError(f'token count mismatch: {folder}/{label}')
        rows.append(row)
        previous = usage['prompt_tokens']
    if not 262144 - 1024 <= previous < 262144 - 512 or rows[-1]['timings']['predicted_n'] != 512:
        raise RuntimeError(f'256K final generation did not complete: {folder}')
    prefill_ms = sum(r['timings']['prompt_ms'] for r in rows)
    tools = rows[1:]
    decode_ms = sum(r['timings']['predicted_ms'] for r in tools)
    decoded = sum(r['timings']['predicted_n'] for r in tools)
    # A fixed input history excludes prior generated text. Count increments of
    # API prompt length, not prompt_n (which can also include checkpoint replay).
    new_positions = sum(r['new_prompt_positions'] for r in rows)
    # run_prefill_multimodal uses eval_end = prompt.tokens.size() - 1 with
    # MTP enabled. Its final pending prompt token is handled in spec_generate.
    new_input_rows = new_positions - 1
    rss = list(csv.DictReader((folder / 'rss.csv').open(encoding='utf-8')))
    runtime = [r for r in rss if r['phase'] != 'loading']
    result = {'request_count': len(rows), 'tool_rounds': len(tools),
              'final_prompt_tokens': previous, 'final_total_tokens': previous + rows[-1]['timings']['predicted_n'],
              'new_prompt_positions': new_positions, 'new_input_rows': new_input_rows,
              'task_prefill_s': prefill_ms / 1000,
              'effective_prefill_tps': new_input_rows * 1000 / prefill_ms,
              'tool_generated_tokens': decoded, 'tool_decode_s': decode_ms / 1000,
              'aggregate_tool_decode_tps': decoded * 1000 / decode_ms,
              'final_decode_tps': rows[-1]['timings']['predicted_per_second'],
              'final_prefill_s': rows[-1]['timings']['prompt_ms'] / 1000,
              'final_ttft_s': rows[-1]['ttft_s'],
              'total_request_wall_s': sum(r['wall_s'] for r in rows),
              'peak_vram_mib': summary['peak_vram_mib'], 'peak_runtime_rss_mib': summary['peak_runtime_rss_mib'],
              'peak_runtime_private_mib': max(float(r['private_mib']) for r in runtime) if runtime and 'private_mib' in runtime[0] else None,
              'sampling_errors': summary['sampling_errors'], 'max_sample_gap_ms': summary['max_sample_gap_ms']}
    # Compare the same context bands, using total tokens / total time.
    result['context_bands'] = []
    for low, high in ((0, 65536), (65536, 131072), (131072, 196608), (196608, 262144)):
        band = [r for r in tools if low < r['prompt_tokens'] <= high]
        result['context_bands'].append({'range': [low, high], 'requests': len(band),
            'new_prompt_positions': sum(r['new_prompt_positions'] for r in band),
            'effective_prefill_tps': 1000 * sum(r['new_prompt_positions'] for r in band) / sum(r['timings']['prompt_ms'] for r in band),
            'decode_tps': 1000 * sum(r['timings']['predicted_n'] for r in band) / sum(r['timings']['predicted_ms'] for r in band)})
    save(folder / 'metrics.json', result)
    save(folder / 'rows.json', rows)
    return rows, result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, default=root / 'build-win/bin/llama-kvmem-server.exe')
    parser.add_argument('--model', type=Path, default=root.parent / 'models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf')
    parser.add_argument('--mmproj', type=Path, default=root.parent / 'models/mmproj-Q8_0.gguf')
    parser.add_argument('--output', type=Path, default=root / 'artifacts/gdn-task2-20261001')
    parser.add_argument('--resume', action='store_true', help='reuse completed groups with identical configuration')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'server': str(args.server.resolve()), 'model': str(args.model.resolve()), 'mmproj': str(args.mmproj.resolve()),
        'server_sha256': digest(args.server), 'patch_sha256': digest(root / 'patches/gdn-output-fusion.patch'),
        'canary_sha256': digest(root / 'scripts/multimodal_canary.py'),
        'task': 'Task 2: 32 x ~8192-token fixed tool results, 256K, thinking 128, output 512',
        'settings': {'ctx': 262144, 'batch': 512, 'budget': 36864, 'reserve': 16384,
                     'mtp': 3, 'mtp_state': 'replay', 'kv': 'q8_0', 'draft_kv': 'f16', 'image_max_tokens': 512},
        'gpu': 'physical NVML index 1, CUDA_VISIBLE_DEVICES=1 with PCI_BUS_ID order',
        'trace': False, 'order': ['baseline', 'fused'], 'complete_runs_per_mode': 1}
    config_path = output / 'config.json'
    if args.resume:
        original = load(config_path)
        changed = [k for k, v in config.items() if k != 'created' and original.get(k) != v]
        if changed:
            parser.error('--resume configuration changed: ' + ', '.join(changed))
    else:
        if config_path.exists():
            parser.error('output already contains a run; use a new output folder or --resume')
        save(config_path, config)
    rows, metrics = {}, {}
    for mode in config['order']:
        folder = output / mode
        folder.mkdir(exist_ok=True)
        completed = False
        if args.resume and (folder / 'summary.json').exists() and (folder / 'long-context.json').exists():
            completed = load(folder / 'long-context.json').get('completed', False)
            launch = load(folder / 'launch.json')
            if launch['fusion'] != ('1' if mode == 'fused' else '0'):
                raise RuntimeError('saved fusion flag differs')
        print(f"{'RESUME' if completed else 'START'} {mode} {folder}", flush=True)
        if not completed:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            cmd = [sys.executable, str(root / 'scripts/multimodal_canary.py'), '--binary', str(args.server.resolve()),
                   '--model', str(args.model.resolve()), '--mmproj', str(args.mmproj.resolve()), '--gpu-index', '1',
                   '--query-replay', 'auto', '--query-policy', 'user', '--mtp-state', 'replay', '--mtp', '3',
                   '--kv', 'q8_0', '--draft-kv', 'f16', '--image-max-tokens', '512', '--budget', '36864',
                   '--reserve', '16384', '--ctx', '262144', '--batch', '512', '--long-context-benchmark',
                   '--long-chunk-tokens', '8192', '--thinking-budget', '128', '--startup-timeout', '900',
                   '--port', str(port), '--folder', str(folder)]
            env = dict(os.environ, PYTHONUTF8='1', KVMEM_GDN_OUT_FUSION='1' if mode == 'fused' else '0',
                       KVMEM_GDN_OUT_FUSION_TRACE='0')
            for key in ('KVMEM_TRACE', 'KVMEM_PERF', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION'):
                env.pop(key, None)
            save(folder / 'launch.json', {'command': cmd, 'fusion': env['KVMEM_GDN_OUT_FUSION']})
            with (folder / 'console.log').open('w', encoding='utf-8') as log:
                subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=log, check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        rows[mode], metrics[mode] = collect(folder)
        save(output / 'partial-summary.json', metrics)
        print('PASS', mode, json.dumps(metrics[mode], ensure_ascii=False), flush=True)
    checks = []
    for off, on in zip(rows['baseline'], rows['fused']):
        checks.append({'label': off['label'],
            'request_identical': off['request_sha256'] == on['request_sha256'],
            'output_identical': all(off[k] == on[k] for k in ('content', 'reasoning_content', 'finish_reason')),
            'usage_identical': off['usage'] == on['usage']})
    changes = {key: metrics['fused'][key] / metrics['baseline'][key] - 1
               for key in ('effective_prefill_tps', 'aggregate_tool_decode_tps', 'final_decode_tps')}
    changes['wall_time_reduction'] = 1 - metrics['fused']['total_request_wall_s'] / metrics['baseline']['total_request_wall_s']
    save(output / 'summary.json', {'metrics': metrics, 'changes': changes, 'checks': checks,
         'all_requests_identical': all(c['request_identical'] for c in checks),
         'all_outputs_identical': all(c['output_identical'] for c in checks),
         'all_usage_identical': all(c['usage_identical'] for c in checks)})
    if not all(c['request_identical'] for c in checks):
        raise RuntimeError('Task 2 input differed; comparison cannot be attributed only to fusion')
    if digest(args.server) != config['server_sha256'] or digest(root / 'patches/gdn-output-fusion.patch') != config['patch_sha256']:
        raise RuntimeError('engine binary or patch changed during experiment')
    print('COMPLETE', json.dumps(changes), flush=True)


if __name__ == '__main__':
    main()
