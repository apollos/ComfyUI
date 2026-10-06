import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import soundfile as sf
import torch

import folder_paths
import comfy.model_management as mm

DEFAULT_INSTRUCTION = '保持说话内容不变，将声音改为十六岁少女：音色清新甜亮，发声清脆轻盈，带着笑意说话，语气活泼俏皮，带一点自然的嗲感和撒娇感。部分句尾的元音轻轻拖长，尾音柔和上扬，像亲昵地向熟悉的人撒娇。整体节奏轻快，吐字清晰，拖尾轻微自然。'


class AuKVoiceEditLocal:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio': ('AUDIO',),
            'instruction': ('STRING', {'multiline': True, 'default': DEFAULT_INSTRUCTION}),
            'steps': ('INT', {'default': 64, 'min': 4, 'max': 64, 'step': 1}),
            'cfg': ('FLOAT', {'default': 2.0, 'min': 0.0, 'max': 5.0, 'step': 0.1}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
        }}

    RETURN_TYPES = ('AUDIO', 'STRING')
    RETURN_NAMES = ('audio', 'run_info')
    FUNCTION = 'edit'
    CATEGORY = 'audio/AuK'
    DESCRIPTION = '通过文字描述转换语音风格。完整音频自动分段后拼接，使用本机 AuK 环境，仅加载已有权重。'

    def edit(self, audio, instruction, steps, cfg, seed):
        python = Path(os.environ.get('AUK_PYTHON', '/home/yu/.venv/auk/bin/python'))
        root = Path(os.environ.get('AUK_ROOT', '/home/yu/Workspace/AuK'))
        if not python.is_file():
            raise FileNotFoundError(f'找不到 AuK 环境：{python}')
        if not instruction.strip():
            raise ValueError('请填写声音描述。')
        waveform = audio['waveform']
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError('AuK 每次接收一段音频，不能使用批量音频。')
        mono = waveform[0].detach().to(device='cpu', dtype=torch.float32).mean(dim=0)
        if not mono.numel() or not torch.isfinite(mono).all():
            raise ValueError('输入音频为空或包含无效数值。')
        runs = Path(folder_paths.get_output_directory()) / 'auk' / 'runs'
        runs.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix='edit-', dir=runs))
        source, result = run / 'input.wav', run / 'result'
        sf.write(source, mono.numpy(), audio['sample_rate'], subtype='FLOAT')
        command = [str(python), str(Path(__file__).with_name('runner.py')),
                   '--root', str(root), '--input', str(source), '--output-dir', str(result),
                   '--instruction', instruction, '--steps', str(steps), '--cfg', str(cfg), '--seed', str(seed)]
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        mm.throw_exception_if_processing_interrupted()
        mm.unload_all_models()
        mm.soft_empty_cache()
        log_path = run / 'run.log'
        print(f'[AuK] 开始转换；运行日志：{log_path}', flush=True)
        with log_path.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                while True:
                    mm.throw_exception_if_processing_interrupted()
                    try:
                        process.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if process.returncode:
            with log_path.open('rb') as log:
                log.seek(max(0, log_path.stat().st_size - 6000))
                tail = log.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'AuK 转换失败，日志：{log_path}\n{tail}')
        data, sample_rate = sf.read(result / 'edited.wav', dtype='float32', always_2d=True)
        generated = torch.from_numpy(data.T.copy()).unsqueeze(0)
        metadata = json.loads((result / 'result.json').read_text())
        info = json.dumps({'directory': str(run), **metadata}, ensure_ascii=False, indent=2)
        print(f'[AuK] 转换完成：{result / "edited.wav"}', flush=True)
        return ({'waveform': generated, 'sample_rate': sample_rate}, info)


NODE_CLASS_MAPPINGS = {'AuKVoiceEditLocal': AuKVoiceEditLocal}
NODE_DISPLAY_NAME_MAPPINGS = {'AuKVoiceEditLocal': 'AuK 语音风格转换（完整音频）'}
