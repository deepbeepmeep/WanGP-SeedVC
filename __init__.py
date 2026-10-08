import os
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torchaudio

from shared.utils import files_locator as fl

# Weight files, modes and defaults are defined by WanGP; this package is the GPL-3.0 SeedVC engine.
from postprocessing.seedvc_processor.assets import (
    SEEDVC_MODE_SPEECH,
    SEEDVC_MODE_SINGING,
    SEEDVC_MODE_ACCENT,
    SEEDVC_ROOT,
    SEEDVC_CAMPPLUS_FILENAME,
    SEEDVC_SPEECH_CHECKPOINT_FILENAME,
    SEEDVC_SPEECH_CONFIG_FILENAME,
    SEEDVC_SINGING_CHECKPOINT_FILENAME,
    SEEDVC_SINGING_CONFIG_FILENAME,
    SEEDVC_RMVPE_FILENAME,
    SEEDVC_V2_AR_CHECKPOINT_FILENAME,
    SEEDVC_V2_CFM_CHECKPOINT_FILENAME,
    SEEDVC_V2_NARROW_CHECKPOINT_FILENAME,
    SEEDVC_V2_WIDE_CHECKPOINT_FILENAME,
    SEEDVC_BIGVGAN_DIR,
    SEEDVC_BIGVGAN_44K_DIR,
    SEEDVC_WHISPER_DIR,
    SEEDVC_HUBERT_DIR,
    normalize_mode,
)

SEEDVC_SAMPLE_RATE = 22050
SEEDVC_MAX_REFERENCE_SECONDS = 25.0
# SeedVC v2 style/AR conversion changes timing, which breaks video remux and speaker masks.
SEEDVC_V2_CONVERT_STYLE = False


def _asset_paths(mode: int | str | None = SEEDVC_MODE_SPEECH, root: str = SEEDVC_ROOT) -> dict[str, str]:
    mode = normalize_mode(mode)
    common = {
        "campplus_path": fl.locate_file(os.path.join(root, SEEDVC_CAMPPLUS_FILENAME)),
        "whisper_folder": fl.locate_folder(SEEDVC_WHISPER_DIR),
    }
    if mode == SEEDVC_MODE_SINGING:
        return {
            **common,
            "checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_SINGING_CHECKPOINT_FILENAME)),
            "config_path": fl.locate_file(os.path.join(root, SEEDVC_SINGING_CONFIG_FILENAME)),
            "rmvpe_path": fl.locate_file(os.path.join(root, SEEDVC_RMVPE_FILENAME)),
            "bigvgan_folder": fl.locate_folder(SEEDVC_BIGVGAN_44K_DIR),
        }
    if mode == SEEDVC_MODE_ACCENT:
        return {
            **common,
            "ar_checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_V2_AR_CHECKPOINT_FILENAME)),
            "cfm_checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_V2_CFM_CHECKPOINT_FILENAME)),
            "narrow_checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_V2_NARROW_CHECKPOINT_FILENAME)),
            "wide_checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_V2_WIDE_CHECKPOINT_FILENAME)),
            "bigvgan_folder": fl.locate_folder(SEEDVC_BIGVGAN_DIR),
            "hubert_folder": fl.locate_folder(SEEDVC_HUBERT_DIR),
        }
    return {
        **common,
        "checkpoint_path": fl.locate_file(os.path.join(root, SEEDVC_SPEECH_CHECKPOINT_FILENAME)),
        "config_path": fl.locate_file(os.path.join(root, SEEDVC_SPEECH_CONFIG_FILENAME)),
        "bigvgan_folder": fl.locate_folder(SEEDVC_BIGVGAN_DIR),
    }


def _closure_modules(fn) -> list[torch.nn.Module]:
    modules = []
    for cell in fn.__closure__ or []:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, torch.nn.Module):
            modules.append(value)
    return modules


def _make_mono(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    return waveform.mean(dim=0, keepdim=True)


def _torch_mono_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    return _make_mono(waveform).squeeze(0).numpy().astype(np.float32, copy=False)


def _save_mono_resampled(path: str, waveform: torch.Tensor, source_rate: int, target_rate: int = SEEDVC_SAMPLE_RATE, max_seconds: float | None = None) -> None:
    import soundfile as sf

    waveform = _make_mono(waveform)
    if int(source_rate) != int(target_rate):
        waveform = torchaudio.functional.resample(waveform, int(source_rate), int(target_rate))
    if max_seconds is not None:
        waveform = waveform[:, : int(round(float(max_seconds) * int(target_rate)))]
    sf.write(path, waveform.squeeze(0).clamp_(-1.0, 1.0).numpy(), int(target_rate))


def _register_unmanaged_seedvc_tensors(modules) -> None:
    for module in modules:
        for submodule in module.modules():
            for attr in ("freqs_cis", "causal_mask", "mask_cache", "input_pos"):
                value = getattr(submodule, attr, None)
                if isinstance(value, torch.Tensor) and attr not in submodule._buffers:
                    delattr(submodule, attr)
                    submodule.register_buffer(attr, value, persistent=False)


def _module_device(module: torch.nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        return tensor.device
    return torch.device("cpu")


def _runtime_device(pipe: dict[str, torch.nn.Module]) -> torch.device:
    for module in pipe.values():
        for submodule in module.modules():
            if hasattr(submodule, "_mm_manager"):
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for module in pipe.values():
        return _module_device(module)
    return torch.device("cpu")


def _normalise_output(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    peak = np.abs(samples).max(initial=0.0)
    return samples / peak if peak > 1.0 else samples


def _audio_tuple_to_stereo_tensor(audio_tuple: tuple[int, np.ndarray], output_rate: int) -> torch.Tensor:
    converted_rate, samples = audio_tuple
    converted_tensor = torch.from_numpy(_normalise_output(samples)).float().unsqueeze(0)
    if int(converted_rate) != int(output_rate):
        converted_tensor = torchaudio.functional.resample(converted_tensor, int(converted_rate), int(output_rate))
    return converted_tensor.repeat(2, 1)


def _consume_generator_return(generator):
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def _configure_pydub_ffmpeg() -> None:
    from shared.utils.video_decode import resolve_media_binary

    ffmpeg_path = resolve_media_binary("ffmpeg")
    ffprobe_path = resolve_media_binary("ffprobe")
    if ffmpeg_path:
        ffmpeg_dir = os.path.dirname(os.fspath(ffmpeg_path))
        if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    from pydub import AudioSegment

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path


def _load_seedvc_app():
    try:
        from . import app_vc
    except ImportError as exc:
        raise ImportError("SeedVC support requires the bundled `postprocessing/seedvc` package files.") from exc
    return app_vc


def _load_seedvc_svc_app():
    try:
        from . import app_svc
    except ImportError as exc:
        raise ImportError("SeedVC singing support requires the bundled `postprocessing/seedvc` package files.") from exc
    return app_svc


class SeedVCVoiceConverter:
    mode = SEEDVC_MODE_SPEECH
    default_steps = 25
    default_cfg_rate = 0.5
    sample_rate = 22050

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        campplus_path: str,
        bigvgan_folder: str,
        whisper_folder: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.checkpoint_path = os.fspath(checkpoint_path)
        self.config_path = os.fspath(config_path)
        self.campplus_path = os.fspath(campplus_path)
        self.bigvgan_folder = os.fspath(bigvgan_folder)
        self.whisper_folder = os.fspath(whisper_folder)
        self.dtype = dtype
        self._app_vc = None
        self._patched_config_path = None
        self._load()

    def _build_local_config(self) -> str:
        import yaml

        with open(self.config_path, "r", encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        config["model_params"]["vocoder"]["name"] = self.bigvgan_folder
        config["model_params"]["speech_tokenizer"]["name"] = self.whisper_folder
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", encoding="utf-8", delete=False)
        with tmp:
            yaml.safe_dump(config, tmp, sort_keys=False)
        self._patched_config_path = tmp.name
        return tmp.name

    def _load(self) -> None:
        _configure_pydub_ffmpeg()
        app_vc = _load_seedvc_app()
        app_vc.device = torch.device("cpu")
        app_vc.load_custom_model_from_hf = self._load_custom_model_from_local_assets
        os.environ.setdefault("HF_HUB_CACHE", str(Path(self.campplus_path).parent / "hf_cache"))
        args = Namespace(checkpoint=self.checkpoint_path, config=self._build_local_config(), fp16=self.dtype == torch.float16, gpu=0)
        (
            app_vc.model,
            app_vc.semantic_fn,
            app_vc.vocoder_fn,
            app_vc.campplus_model,
            app_vc.to_mel,
            app_vc.mel_fn_args,
        ) = app_vc.load_models(args)
        app_vc.max_context_window = app_vc.sr // app_vc.hop_length * 30
        app_vc.overlap_wave_len = app_vc.overlap_frame_len * app_vc.hop_length
        self._app_vc = app_vc

        self.seedvc_model = torch.nn.ModuleDict({str(name): module for name, module in app_vc.model.items() if isinstance(module, torch.nn.Module)})
        self.semantic_modules = torch.nn.ModuleList(_closure_modules(app_vc.semantic_fn))
        self.campplus_model = app_vc.campplus_model
        self.vocoder_fn = app_vc.vocoder_fn
        _register_unmanaged_seedvc_tensors(self.pipe_modules().values())
        for module in self.pipe_modules().values():
            for submodule in module.modules():
                submodule._lock_dtype = None

    def pipe_modules(self) -> dict[str, torch.nn.Module]:
        pipe = {f"seedvc_{name}": module for name, module in self.seedvc_model.items()}
        if len(self.semantic_modules) == 1:
            pipe["seedvc_whisper_small"] = self.semantic_modules[0]
        else:
            pipe.update({f"seedvc_speech_tokenizer_{idx + 1}": module for idx, module in enumerate(self.semantic_modules)})
        if isinstance(self.campplus_model, torch.nn.Module):
            pipe["seedvc_campplus"] = self.campplus_model
        if isinstance(self.vocoder_fn, torch.nn.Module):
            pipe["seedvc_bigvgan"] = self.vocoder_fn
        return pipe

    def _load_custom_model_from_local_assets(self, repo_id, model_filename, config_filename=None):
        if repo_id == "funasr/campplus" and model_filename == SEEDVC_CAMPPLUS_FILENAME:
            return self.campplus_path
        raise FileNotFoundError(f"SeedVC asset is not declared for local loading: {repo_id}/{model_filename}")

    def forward(
        self,
        source_wav_path: str,
        target_wav_path: str,
        diffusion_steps: int | None = None,
        cfg_rate: float | None = None,
    ) -> tuple[np.ndarray, int]:
        if self._app_vc is None:
            raise RuntimeError("SeedVC is not loaded.")
        _configure_pydub_ffmpeg()
        self._app_vc.device = _runtime_device(self.pipe_modules())
        audio_tuple = None
        for result in self._app_vc.voice_conversion(
            source=source_wav_path,
            target=target_wav_path,
            diffusion_steps=self.default_steps if diffusion_steps is None else int(diffusion_steps),
            length_adjust=1.0,
            inference_cfg_rate=self.default_cfg_rate if cfg_rate is None else float(cfg_rate),
        ):
            if isinstance(result, tuple) and len(result) == 2:
                _, audio_tuple = result
        if audio_tuple is None:
            raise RuntimeError("SeedVC produced no output.")
        sample_rate, samples = audio_tuple
        return int(sample_rate), _normalise_output(samples)

    def convert_tensor(
        self,
        source_audio: torch.Tensor,
        source_rate: int,
        reference_audio: torch.Tensor,
        reference_rate: int,
        output_rate: int,
        diffusion_steps: int | None = None,
        cfg_rate: float | None = None,
    ) -> torch.Tensor:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source_22k.wav")
            target_path = os.path.join(tmpdir, "target_22k.wav")
            _save_mono_resampled(source_path, source_audio, source_rate, target_rate=self.sample_rate)
            _save_mono_resampled(target_path, reference_audio, reference_rate, target_rate=self.sample_rate, max_seconds=SEEDVC_MAX_REFERENCE_SECONDS)
            converted = self.forward(source_path, target_path, diffusion_steps=diffusion_steps, cfg_rate=cfg_rate)
        return _audio_tuple_to_stereo_tensor(converted, output_rate)


class SeedVCSingingConverter(SeedVCVoiceConverter):
    mode = SEEDVC_MODE_SINGING
    default_steps = 10
    default_cfg_rate = 0.7
    sample_rate = 44100

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        campplus_path: str,
        rmvpe_path: str,
        bigvgan_folder: str,
        whisper_folder: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.rmvpe_path = os.fspath(rmvpe_path)
        super().__init__(checkpoint_path, config_path, campplus_path, bigvgan_folder, whisper_folder, dtype=dtype)

    def _load(self) -> None:
        _configure_pydub_ffmpeg()
        app_svc = _load_seedvc_svc_app()
        app_svc.device = torch.device("cpu")
        app_svc.load_custom_model_from_hf = self._load_custom_model_from_local_assets
        os.environ.setdefault("HF_HUB_CACHE", str(Path(self.campplus_path).parent / "hf_cache"))
        args = Namespace(checkpoint=self.checkpoint_path, config=self._build_local_config(), fp16=self.dtype == torch.float16, gpu=0)
        (
            app_svc.model_f0,
            app_svc.semantic_fn,
            app_svc.vocoder_fn,
            app_svc.campplus_model,
            app_svc.to_mel_f0,
            app_svc.mel_fn_args,
            app_svc.f0_fn,
        ) = app_svc.load_models(args)
        app_svc.max_context_window = app_svc.sr // app_svc.hop_length * 30
        app_svc.overlap_wave_len = app_svc.overlap_frame_len * app_svc.hop_length
        self._app_vc = app_svc

        self.seedvc_model = torch.nn.ModuleDict({str(name): module for name, module in app_svc.model_f0.items() if isinstance(module, torch.nn.Module)})
        self.semantic_modules = torch.nn.ModuleList(_closure_modules(app_svc.semantic_fn))
        self.campplus_model = app_svc.campplus_model
        self.vocoder_fn = app_svc.vocoder_fn
        self.f0_extractor = getattr(app_svc.f0_fn, "__self__", None)
        _register_unmanaged_seedvc_tensors(self.pipe_modules().values())
        for module in self.pipe_modules().values():
            for submodule in module.modules():
                submodule._lock_dtype = None

    def _load_custom_model_from_local_assets(self, repo_id, model_filename, config_filename=None):
        if repo_id == "funasr/campplus" and model_filename == SEEDVC_CAMPPLUS_FILENAME:
            return self.campplus_path
        if repo_id == "lj1995/VoiceConversionWebUI" and model_filename == SEEDVC_RMVPE_FILENAME:
            return self.rmvpe_path
        raise FileNotFoundError(f"SeedVC singing asset is not declared for local loading: {repo_id}/{model_filename}")

    def pipe_modules(self) -> dict[str, torch.nn.Module]:
        pipe = super().pipe_modules()
        if self.f0_extractor is not None:
            for attr in ("mel_extractor", "model"):
                module = getattr(self.f0_extractor, attr, None)
                if isinstance(module, torch.nn.Module):
                    pipe[f"seedvc_f0_{attr}"] = module
        return pipe

    def forward(
        self,
        source_wav_path: str,
        target_wav_path: str,
        diffusion_steps: int | None = None,
        cfg_rate: float | None = None,
    ) -> tuple[np.ndarray, int]:
        if self._app_vc is None:
            raise RuntimeError("SeedVC singing model is not loaded.")
        _configure_pydub_ffmpeg()
        self._app_vc.device = _runtime_device(self.pipe_modules())
        if self.f0_extractor is not None:
            self.f0_extractor.device = self._app_vc.device
        audio_tuple = None
        for result in self._app_vc.voice_conversion(
            source=source_wav_path,
            target=target_wav_path,
            diffusion_steps=self.default_steps if diffusion_steps is None else int(diffusion_steps),
            length_adjust=1.0,
            inference_cfg_rate=self.default_cfg_rate if cfg_rate is None else float(cfg_rate),
            auto_f0_adjust=True,
            pitch_shift=0,
        ):
            if isinstance(result, tuple) and len(result) == 2:
                _, audio_tuple = result
        if audio_tuple is None:
            raise RuntimeError("SeedVC singing model produced no output.")
        sample_rate, samples = audio_tuple
        return int(sample_rate), _normalise_output(samples)


class SeedVCAccentConverter:
    mode = SEEDVC_MODE_ACCENT
    default_steps = 30
    default_cfg_rate = 0.7
    sample_rate = 22050

    def __init__(
        self,
        ar_checkpoint_path: str,
        cfm_checkpoint_path: str,
        narrow_checkpoint_path: str,
        wide_checkpoint_path: str,
        campplus_path: str,
        bigvgan_folder: str,
        whisper_folder: str,
        hubert_folder: str,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.ar_checkpoint_path = os.fspath(ar_checkpoint_path)
        self.cfm_checkpoint_path = os.fspath(cfm_checkpoint_path)
        self.narrow_checkpoint_path = os.fspath(narrow_checkpoint_path)
        self.wide_checkpoint_path = os.fspath(wide_checkpoint_path)
        self.campplus_path = os.fspath(campplus_path)
        self.bigvgan_folder = os.fspath(bigvgan_folder)
        self.whisper_folder = os.fspath(whisper_folder)
        self.hubert_folder = os.fspath(hubert_folder)
        self.dtype = dtype
        self.vc_wrapper = None
        self._patched_config_path = None
        self._load()

    def _build_local_config(self) -> str:
        import yaml

        config_path = Path(__file__).resolve().parent / "configs" / "v2" / "vc_wrapper.yaml"
        with open(config_path, "r", encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        config["vocoder"]["pretrained_model_name_or_path"] = self.bigvgan_folder
        for key in ("content_extractor_narrow", "content_extractor_wide"):
            config[key]["tokenizer_name"] = self.whisper_folder
            config[key]["ssl_model_name"] = self.hubert_folder
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False)
        with tmp:
            yaml.safe_dump(config, tmp, sort_keys=False)
        self._patched_config_path = tmp.name
        return tmp.name

    def _load(self) -> None:
        import yaml
        from hydra.utils import instantiate
        from omegaconf import DictConfig

        _configure_pydub_ffmpeg()
        from .modules.v2 import vc_wrapper as vc_wrapper_module

        vc_wrapper_module.load_custom_model_from_hf = self._load_custom_model_from_local_assets
        os.environ.setdefault("HF_HUB_CACHE", str(Path(self.campplus_path).parent / "hf_cache"))
        with open(self._build_local_config(), "r", encoding="utf-8") as reader:
            cfg = DictConfig(yaml.safe_load(reader))
        self.vc_wrapper = instantiate(cfg)
        self.vc_wrapper.load_checkpoints(ar_checkpoint_path=self.ar_checkpoint_path, cfm_checkpoint_path=self.cfm_checkpoint_path)
        self.vc_wrapper.to(torch.device("cpu"))
        self.vc_wrapper.eval()
        self.vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=self.dtype, device=torch.device("cpu"))
        _register_unmanaged_seedvc_tensors(self.pipe_modules().values())
        for module in self.pipe_modules().values():
            for submodule in module.modules():
                submodule._lock_dtype = None

    def _load_custom_model_from_local_assets(self, repo_id, model_filename, config_filename=None):
        if repo_id == "Plachta/ASTRAL-quantization" and model_filename == SEEDVC_V2_NARROW_CHECKPOINT_FILENAME:
            return self.narrow_checkpoint_path
        if repo_id == "Plachta/ASTRAL-quantization" and model_filename == SEEDVC_V2_WIDE_CHECKPOINT_FILENAME:
            return self.wide_checkpoint_path
        if repo_id == "funasr/campplus" and model_filename == SEEDVC_CAMPPLUS_FILENAME:
            return self.campplus_path
        raise FileNotFoundError(f"SeedVC v2 asset is not declared for local loading: {repo_id}/{model_filename}")

    def pipe_modules(self) -> dict[str, torch.nn.Module]:
        if self.vc_wrapper is None:
            return {}
        return {f"seedvc_v2_{name}": module for name, module in self.vc_wrapper.named_children() if isinstance(module, torch.nn.Module)}

    def convert_tensor(
        self,
        source_audio: torch.Tensor,
        source_rate: int,
        reference_audio: torch.Tensor,
        reference_rate: int,
        output_rate: int,
        diffusion_steps: int | None = None,
        cfg_rate: float | None = None,
    ) -> torch.Tensor:
        if self.vc_wrapper is None:
            raise RuntimeError("SeedVC v2 model is not loaded.")
        device = _runtime_device(self.pipe_modules())
        dtype = self.dtype if device.type == "cuda" else torch.float32
        generator = self.vc_wrapper.convert_voice_arrays(
            source_wave=_torch_mono_to_numpy(source_audio),
            target_wave=_torch_mono_to_numpy(reference_audio),
            source_sr=int(source_rate),
            target_sr=int(reference_rate),
            diffusion_steps=self.default_steps if diffusion_steps is None else int(diffusion_steps),
            length_adjust=1.0,
            intelligebility_cfg_rate=self.default_cfg_rate if cfg_rate is None else float(cfg_rate),
            similarity_cfg_rate=self.default_cfg_rate if cfg_rate is None else float(cfg_rate),
            top_p=0.9,
            temperature=1.0,
            repetition_penalty=1.0,
            convert_style=SEEDVC_V2_CONVERT_STYLE,
            anonymization_only=False,
            device=device,
            dtype=dtype,
        )
        audio_tuple = _consume_generator_return(generator)
        if audio_tuple is None:
            raise RuntimeError("SeedVC v2 produced no output.")
        return _audio_tuple_to_stereo_tensor(audio_tuple, output_rate)


def get_model(dtype: torch.dtype = torch.float16, root: str = SEEDVC_ROOT, mode: int | str | None = SEEDVC_MODE_SPEECH):
    mode = normalize_mode(mode)
    converter_cls = {
        SEEDVC_MODE_SPEECH: SeedVCVoiceConverter,
        SEEDVC_MODE_SINGING: SeedVCSingingConverter,
        SEEDVC_MODE_ACCENT: SeedVCAccentConverter,
    }[mode]
    return converter_cls(**_asset_paths(mode, root), dtype=dtype)


def get_pipe(profile_no=None, dtype: torch.dtype = torch.float16, root: str = SEEDVC_ROOT, model=None, mode: int | str | None = SEEDVC_MODE_SPEECH) -> dict[str, torch.nn.Module]:
    seedvc_model = get_model(dtype=dtype, root=root, mode=mode) if model is None else model
    return seedvc_model.pipe_modules()


def get_cotenants_map(pipe: dict[str, torch.nn.Module]) -> dict[str, list[str]]:
    seedvc_keys = [key for key in pipe if str(key).startswith("seedvc_")]
    return {key: list(seedvc_keys) for key in seedvc_keys}
