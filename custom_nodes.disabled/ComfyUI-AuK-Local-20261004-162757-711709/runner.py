#!/usr/bin/env python3
"""Offline AuK Base speech editing; no download, pip, git, or cloud calls."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/home/yu/Workspace/AuK')
PROMPT = '保持说话内容不变，将声音改为十六岁少女：音色清新甜亮，发声清脆轻盈，带着笑意说话，语气活泼俏皮，带一点自然的嗲感和撒娇感。部分句尾的元音轻轻拖长，尾音柔和上扬，像亲昵地向熟悉的人撒娇。整体节奏轻快，吐字清晰，拖尾轻微自然。'
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'TORCHDYNAMO_DISABLE'):
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True


def block_network(*args, **kwargs):
    raise RuntimeError('本测试禁止联网；缺少模型时请使用单独的 ModelScope 下载命令。')


socket.create_connection = block_network
for method in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
    if hasattr(socket.socket, method):
        setattr(socket.socket, method, block_network)


def check_models(root):
    base, qwen = root / 'ckpts/AuK', root / 'ckpts/Qwen2.5-Omni-3B'
    needed = [base / f for f in ('config.yaml', 'auk_base.safetensors', 'vae.safetensors')]
    needed += [qwen / f for f in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json')]
    index = qwen / 'model.safetensors.index.json'
    if index.is_file():
        names = set(json.loads(index.read_text())['weight_map'].values())
        for name in names:
            p = qwen / name
            if not p.resolve().is_relative_to(qwen.resolve()):
                raise ValueError('无效模型分片路径：' + name)
            needed.append(p)
    else:
        needed.append(qwen / 'model.safetensors')
    missing = [str(p) for p in needed if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('本地权重或配置缺失，尚未加载模型：\n' + '\n'.join(missing)
                                + '\n请先执行已提供的 ModelScope 下载命令；本脚本不会下载。')
    from safetensors import safe_open
    for p in needed:
        if p.suffix == '.safetensors':
            with safe_open(str(p), framework='pt', device='cpu') as f:
                if not list(f.keys()):
                    raise ValueError('空权重文件：' + str(p))
    print('MODEL_FILES_OK：必要文件及 safetensors 文件头可读；尚未验证完整权重与推理。')
    return base, qwen


def split_ranges(mono, sr):
    """Cover every source sample exactly once; choose quiet boundaries <=14s.

    Each equal-duration source/target pair stays below the ComfyUI 30s budget.
    Boundary search uses 20ms RMS windows; it does not trim or remove silence.
    """
    import numpy as np
    start, ranges = 0, []
    limit = int(14 * sr)
    window = max(1, int(0.02 * sr))
    while len(mono) - start > limit:
        remaining = len(mono) - start
        nominal = start + int(min(12 * sr, remaining / 2))
        low = max(start + sr, nominal - sr)
        high = min(start + limit - window, nominal + sr, len(mono) - sr)
        candidates = range(low, high + 1, window)
        cut = min(candidates, key=lambda c: float(np.mean(mono[c:c+window].astype(np.float64) ** 2)))
        cut += window // 2
        ranges.append((start, cut))
        start = cut
    ranges.append((start, len(mono)))
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--seconds', type=float, default=None, help='默认读取从起点到文件结尾的全部音频；仅显式指定时截取。')
    parser.add_argument('--instruction', default=PROMPT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--cfg', type=float, default=2.0, help='AuK Base 引导强度，0～5；较高不保证音质或风格更好。')
    parser.add_argument('--check-models', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.cfg) or not 0 <= args.cfg <= 5:
        parser.error('--cfg 必须在 0～5 之间。')
    base, qwen = check_models(args.root)
    if args.check_models:
        return
    if not args.input or not args.input.is_file():
        parser.error('请用 --input 指定本地音频文件。')
    if not math.isfinite(args.start) or args.start < 0 or (args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0)):
        parser.error('--start 必须 >=0，指定 --seconds 时必须 >0。')
    if not args.instruction.strip() or not 4 <= args.steps <= 64:
        parser.error('提示词不能为空，采样步数需在 4～64。')
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('未找到 ROCm GPU，请在本机终端运行。')
    with sf.SoundFile(args.input) as f:
        sr = f.samplerate
        offset = round(args.start * sr)
        if offset >= len(f):
            raise ValueError('起始时间超过音频长度。')
        f.seek(offset)
        samples = f.read(-1 if args.seconds is None else round(args.seconds * sr), dtype='float32', always_2d=True)
    mono = samples.mean(axis=1)
    if len(mono) < sr * 0.5 or not np.isfinite(mono).all() or np.max(np.abs(mono)) < 1e-6:
        raise ValueError('片段过短、无效或静音，请调整起始时间。')
    seconds = len(mono) / sr
    ranges = split_ranges(mono, sr)
    print(f'使用全部选定音频：{seconds:.3f} 秒；转换 {len(ranges)} 段，无丢弃、无重叠。')
    outdir = args.output_dir or args.root / 'outputs' / datetime.now().astimezone().strftime('edit-%Y%m%d-%H%M%S-%f')
    outdir.mkdir(parents=True, exist_ok=False)
    sf.write(outdir / 'source.wav', mono, sr, subtype='FLOAT')
    metadata = {'status': 'loading', 'input': str(args.input.resolve()), 'start': args.start,
                'seconds': seconds, 'instruction': args.instruction, 'seed': args.seed,
                'steps': args.steps, 'cfg': args.cfg, 'cpu_offload': True,
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                'segments': [{'source_start_frame': a, 'source_end_frame': b,
                              'source_seconds': (b-a)/sr} for a, b in ranges]}
    report = outdir / 'result.json'
    def save_report():
        report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    save_report()
    try:
        from auk.infer.infer_auk import AukInfer
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        engine = AukInfer(str(base / 'config.yaml'), str(base / 'auk_base.safetensors'),
                          device='cuda:0', dtype='bf16', qwen_path=str(qwen), cpu_offload=True)
        torch.cuda.synchronize()
        metadata['load_seconds'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        parts = []
        output_sr = None
        for i, (a, b) in enumerate(ranges):
            print(f'转换第 {i+1}/{len(ranges)} 段：{a/sr:.3f}～{b/sr:.3f} 秒', flush=True)
            waveform = torch.from_numpy(mono[a:b].copy()).unsqueeze(0)
            qwen_audio = torchaudio.functional.resample(waveform, sr, 16000).squeeze(0).numpy()
            messages = [{'role': 'user', 'content': [
                {'type': 'text', 'text': args.instruction}, {'type': 'audio', 'audio': qwen_audio}]}]
            torch.manual_seed(args.seed)
            segment, segment_sr = engine.generate(messages, audio=(waveform, sr), gen_seconds=(b-a)/sr,
                                                  nfe=args.steps, cfg_strength=args.cfg, seed=args.seed)
            if output_sr is not None and output_sr != segment_sr:
                raise ValueError('分段输出采样率不一致。')
            output_sr = segment_sr
            segment = segment.detach().float().cpu().reshape(-1).numpy()
            if not len(segment) or not np.isfinite(segment).all():
                raise ValueError('分段输出为空或含非有限值。')
            sf.write(outdir / f'part-{i+1:02d}.wav', segment, output_sr, subtype='FLOAT')
            metadata['segments'][i]['output_seconds'] = len(segment) / output_sr
            parts.append(segment)
            save_report()
        torch.cuda.synchronize()
        metadata['inference_seconds'] = time.perf_counter() - t0
        output = np.concatenate(parts)
        if not len(output) or not np.isfinite(output).all():
            raise ValueError('输出为空或含非有限值。')
        sf.write(outdir / 'edited.wav', output, output_sr, subtype='FLOAT')
        metadata.update(status='generated', output_seconds=len(output)/output_sr,
                        peak_amplitude=float(np.max(np.abs(output))),
                        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        print('生成完成，请对比试听：', outdir / 'source.wav', outdir / 'edited.wav', sep='\n')
    except Exception as e:
        metadata.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        save_report()
        print('运行记录：', report)


if __name__ == '__main__':
    main()
