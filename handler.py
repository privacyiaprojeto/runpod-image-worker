import base64
import io
import os
import shutil
import time
import traceback
import urllib.request
from typing import Any, Dict, Optional

# ============================================================
# Cache/volume config precisa vir antes de carregar o modelo
# ============================================================

CACHE_ROOT = os.getenv("HF_HOME", "/runpod-volume/huggingface").strip() or "/runpod-volume/huggingface"
HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_ROOT, "hub")).strip()
TMP_ROOT = os.getenv("TMPDIR", "/runpod-volume/tmp").strip() or "/runpod-volume/tmp"

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(HUB_CACHE, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)

os.environ["HF_HOME"] = CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", CACHE_ROOT)
os.environ["DIFFUSERS_CACHE"] = os.getenv("DIFFUSERS_CACHE", CACHE_ROOT)
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ["TMPDIR"] = TMP_ROOT

import runpod
import torch
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
)


MODEL_ID = os.getenv("MODEL_ID", "RunDiffusion/Juggernaut-XL-v9")
MODEL_VARIANT = (os.getenv("MODEL_VARIANT", "fp16").strip() or None)

DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "25"))
DEFAULT_GUIDANCE_SCALE = float(os.getenv("DEFAULT_GUIDANCE_SCALE", "6.5"))
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "1216"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "1216"))
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "png").lower().strip()
USE_CPU_OFFLOAD = os.getenv("USE_CPU_OFFLOAD", "false").lower().strip() == "true"

# ============================================================
# Worker V2 - identidade visual / controle opcional
# ============================================================

USE_IP_ADAPTER = os.getenv("USE_IP_ADAPTER", "true").lower().strip() == "true"
IP_ADAPTER_STRICT = os.getenv("IP_ADAPTER_STRICT", "false").lower().strip() == "true"
IP_ADAPTER_REPO = os.getenv("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_SUBFOLDER = os.getenv("IP_ADAPTER_SUBFOLDER", "sdxl_models")
IP_ADAPTER_WEIGHT_NAME = os.getenv("IP_ADAPTER_WEIGHT_NAME", "ip-adapter-plus_sdxl_vit-h.safetensors")
IP_ADAPTER_IMAGE_ENCODER_FOLDER = os.getenv("IP_ADAPTER_IMAGE_ENCODER_FOLDER", "models/image_encoder")
IP_ADAPTER_SCALE = float(os.getenv("IP_ADAPTER_SCALE", "0.65"))

USE_CONTROLNET_OPENPOSE = os.getenv("USE_CONTROLNET_OPENPOSE", "false").lower().strip() == "true"
CONTROLNET_STRICT = os.getenv("CONTROLNET_STRICT", "false").lower().strip() == "true"
CONTROLNET_OPENPOSE_MODEL_ID = os.getenv("CONTROLNET_OPENPOSE_MODEL_ID", "thibaud/controlnet-openpose-sdxl-1.0")
CONTROLNET_CONDITIONING_SCALE = float(os.getenv("CONTROLNET_CONDITIONING_SCALE", "0.8"))

MAX_REFERENCE_IMAGE_BYTES = int(os.getenv("MAX_REFERENCE_IMAGE_BYTES", str(12 * 1024 * 1024)))
REFERENCE_IMAGE_TIMEOUT_SECONDS = int(os.getenv("REFERENCE_IMAGE_TIMEOUT_SECONDS", "45"))

DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "DEFAULT_NEGATIVE_PROMPT",
    "low quality, blurry, distorted face, bad anatomy, extra fingers, missing fingers, deformed hands, watermark, text, logo, duplicate person, oversaturated"
)

TEXT_PIPE = None
CONTROLNET_PIPE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def _log(message: str) -> None:
    print(message, flush=True)


def _disk_report(path: str) -> str:
    try:
        usage = shutil.disk_usage(path)
        total_gb = round(usage.total / (1024 ** 3), 2)
        free_gb = round(usage.free / (1024 ** 3), 2)
        used_gb = round(usage.used / (1024 ** 3), 2)
        return f"path={path} total_gb={total_gb} used_gb={used_gb} free_gb={free_gb}"
    except Exception as exc:
        return f"path={path} disk_report_error={exc}"


def _sanitize_prompt(value: Any, max_len: int = 2200) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text[:max_len]


def _clamp_dimension(value: Any, default: int, max_value: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = default

    number = max(512, min(number, max_value))
    return int(round(number / 8) * 8)


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        number = int(value)
    except Exception:
        number = default

    return max(min_value, min(number, max_value))


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = default

    return max(min_value, min(number, max_value))


def _read_first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "runpod-image-worker/0.2.0"},
    )

    with urllib.request.urlopen(request, timeout=REFERENCE_IMAGE_TIMEOUT_SECONDS) as response:
        data = response.read(MAX_REFERENCE_IMAGE_BYTES + 1)

    if len(data) > MAX_REFERENCE_IMAGE_BYTES:
        raise ValueError("imagem de referência excedeu limite configurado")

    return data


def _decode_base64_image(value: str) -> bytes:
    raw = str(value or "").strip()

    if raw.startswith("data:") and ";base64," in raw:
        raw = raw.split(";base64,", 1)[1]

    raw = raw.replace("base64,", "").replace("\n", "").replace("\r", "").strip()
    return base64.b64decode(raw)


def _open_image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image


def _resize_condition_image(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.LANCZOS).convert("RGB")


def _extract_image(payload: Dict[str, Any], *, url_keys, base64_keys, label: str) -> Optional[Image.Image]:
    for key in base64_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return _open_image_from_bytes(_decode_base64_image(value))
            except Exception as exc:
                _log(f"[JOB] falha ao ler {label} base64 key={key}: {exc}")
                raise

    for key in url_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return _open_image_from_bytes(_download_bytes(value.strip()))
            except Exception as exc:
                _log(f"[JOB] falha ao baixar {label} url key={key}: {exc}")
                raise

    return None


def _extract_identity_image(payload: Dict[str, Any]) -> Optional[Image.Image]:
    companion = payload.get("companion") if isinstance(payload.get("companion"), dict) else {}

    identity_url = _read_first_string(
        payload.get("identity_image_url"),
        payload.get("identityImageUrl"),
        payload.get("reference_image_url"),
        payload.get("referenceImageUrl"),
        companion.get("identity_image_url"),
        companion.get("reference_image_url"),
        companion.get("avatar_url"),
        companion.get("thumbnail_url"),
        companion.get("banner_url"),
    )

    identity_base64 = _read_first_string(
        payload.get("identity_image_base64"),
        payload.get("identityImageBase64"),
        payload.get("reference_image_base64"),
        payload.get("referenceImageBase64"),
    )

    return _extract_image(
        {"identity_url": identity_url, "identity_base64": identity_base64},
        url_keys=["identity_url"],
        base64_keys=["identity_base64"],
        label="identity_image",
    )


def _extract_pose_image(payload: Dict[str, Any]) -> Optional[Image.Image]:
    return _extract_image(
        payload,
        url_keys=["pose_image_url", "poseImageUrl", "control_image_url", "controlImageUrl"],
        base64_keys=["pose_image_base64", "poseImageBase64", "control_image_base64", "controlImageBase64"],
        label="pose_image",
    )


def _configure_pipeline(pipe, *, use_ip_adapter: bool):
    try:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception as exc:
        _log(f"[BOOT] Scheduler fallback: {exc}")

    if DEVICE == "cuda":
        if USE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
            _log("[BOOT] CPU offload enabled")
        else:
            pipe.to("cuda")

        try:
            pipe.enable_xformers_memory_efficient_attention()
            _log("[BOOT] xformers memory efficient attention enabled")
        except Exception as exc:
            _log(f"[BOOT] xformers unavailable, continuing without it: {exc}")
    else:
        pipe.to("cpu")
        _log("[BOOT] CUDA unavailable, running on CPU")

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    pipe._identity_adapter_ready = False

    if use_ip_adapter and USE_IP_ADAPTER:
        try:
            _log(
                f"[BOOT] Loading IP-Adapter repo={IP_ADAPTER_REPO} "
                f"subfolder={IP_ADAPTER_SUBFOLDER} weight={IP_ADAPTER_WEIGHT_NAME} "
                f"image_encoder_folder={IP_ADAPTER_IMAGE_ENCODER_FOLDER} scale={IP_ADAPTER_SCALE}"
            )
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=IP_ADAPTER_WEIGHT_NAME,
                image_encoder_folder=IP_ADAPTER_IMAGE_ENCODER_FOLDER,
            )
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            pipe._identity_adapter_ready = True
            _log("[BOOT] IP-Adapter ready")
        except Exception as exc:
            _log(f"[BOOT] IP-Adapter unavailable: {exc}")
            if IP_ADAPTER_STRICT:
                raise

    return pipe


def _log_boot_common(start: float, mode: str) -> None:
    _log(f"[BOOT] Starting lazy model load mode={mode}")
    _log(f"[BOOT] MODEL_ID={MODEL_ID}")
    _log(f"[BOOT] MODEL_VARIANT={MODEL_VARIANT}")
    _log(f"[BOOT] DEVICE={DEVICE}")
    _log(f"[BOOT] USE_IP_ADAPTER={USE_IP_ADAPTER}")
    _log(f"[BOOT] USE_CONTROLNET_OPENPOSE={USE_CONTROLNET_OPENPOSE}")
    _log(f"[BOOT] HF_HOME={os.environ.get('HF_HOME')}")
    _log(f"[BOOT] HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE')}")
    _log(f"[BOOT] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
    _log(f"[BOOT] DIFFUSERS_CACHE={os.environ.get('DIFFUSERS_CACHE')}")
    _log(f"[BOOT] TMPDIR={os.environ.get('TMPDIR')}")
    _log(f"[BOOT] DISK_CACHE_ROOT {_disk_report(CACHE_ROOT)}")
    _log(f"[BOOT] DISK_TMP_ROOT {_disk_report(TMP_ROOT)}")


def load_text_pipeline(*, use_ip_adapter: bool):
    global TEXT_PIPE

    if TEXT_PIPE is not None:
        return TEXT_PIPE

    start = time.time()
    _log_boot_common(start, "text2image")

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        variant=MODEL_VARIANT,
        use_safetensors=True,
    )

    TEXT_PIPE = _configure_pipeline(pipe, use_ip_adapter=use_ip_adapter)
    _log(f"[BOOT] Text image model ready in {round(time.time() - start, 2)}s")
    return TEXT_PIPE


def load_controlnet_pipeline(*, use_ip_adapter: bool):
    global CONTROLNET_PIPE

    if CONTROLNET_PIPE is not None:
        return CONTROLNET_PIPE

    start = time.time()
    _log_boot_common(start, "controlnet_openpose")
    _log(f"[BOOT] CONTROLNET_OPENPOSE_MODEL_ID={CONTROLNET_OPENPOSE_MODEL_ID}")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_OPENPOSE_MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        use_safetensors=True,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=TORCH_DTYPE,
        variant=MODEL_VARIANT,
        use_safetensors=True,
    )

    CONTROLNET_PIPE = _configure_pipeline(pipe, use_ip_adapter=use_ip_adapter)
    _log(f"[BOOT] ControlNet image model ready in {round(time.time() - start, 2)}s")
    return CONTROLNET_PIPE


def _run_generation_with_identity_fallback(pipe, generation_kwargs: Dict[str, Any], *, identity_enabled: bool):
    try:
        return pipe(**generation_kwargs).images[0], identity_enabled, None
    except RuntimeError as exc:
        message = str(exc)
        is_ip_adapter_shape_error = "mat1 and mat2 shapes cannot be multiplied" in message

        if identity_enabled and is_ip_adapter_shape_error and not IP_ADAPTER_STRICT:
            _log(
                "[JOB] IP-Adapter runtime shape mismatch; "
                "retrying once without identity adapter because IP_ADAPTER_STRICT=false"
            )
            fallback_kwargs = dict(generation_kwargs)
            fallback_kwargs.pop("ip_adapter_image", None)
            return pipe(**fallback_kwargs).images[0], False, message

        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()

    try:
        payload = event.get("input") or {}

        prompt = _sanitize_prompt(payload.get("prompt"))
        if not prompt:
            return {"error": "prompt obrigatório"}

        negative_prompt = _sanitize_prompt(
            payload.get("negative_prompt") or payload.get("negativePrompt") or DEFAULT_NEGATIVE_PROMPT,
            max_len=1800,
        )

        width = _clamp_dimension(payload.get("width"), DEFAULT_WIDTH, MAX_WIDTH)
        height = _clamp_dimension(payload.get("height"), DEFAULT_HEIGHT, MAX_HEIGHT)
        steps = _clamp_int(payload.get("steps") or payload.get("num_inference_steps"), DEFAULT_STEPS, 1, 60)
        guidance_scale = _clamp_float(
            payload.get("guidance_scale") or payload.get("guidanceScale"),
            DEFAULT_GUIDANCE_SCALE,
            1.0,
            12.0,
        )

        if payload.get("seed") is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        else:
            seed = _clamp_int(payload.get("seed"), 0, 0, 2**31 - 1)

        identity_image = None
        identity_enabled = False
        if USE_IP_ADAPTER:
            try:
                identity_image = _extract_identity_image(payload)
                identity_enabled = identity_image is not None
            except Exception:
                if IP_ADAPTER_STRICT:
                    raise
                identity_enabled = False
                identity_image = None

        pose_image = None
        controlnet_enabled = False
        if USE_CONTROLNET_OPENPOSE:
            try:
                pose_image = _extract_pose_image(payload)
                controlnet_enabled = pose_image is not None
            except Exception:
                if CONTROLNET_STRICT:
                    raise
                controlnet_enabled = False
                pose_image = None

        pipe = (
            load_controlnet_pipeline(use_ip_adapter=identity_enabled)
            if controlnet_enabled
            else load_text_pipeline(use_ip_adapter=identity_enabled)
        )

        generator = (
            torch.Generator(device=DEVICE).manual_seed(seed)
            if DEVICE == "cuda"
            else torch.Generator().manual_seed(seed)
        )

        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if identity_enabled and getattr(pipe, "_identity_adapter_ready", False):
            generation_kwargs["ip_adapter_image"] = identity_image
        elif identity_enabled and IP_ADAPTER_STRICT:
            raise RuntimeError("IP-Adapter solicitado, mas não ficou pronto no pipeline")

        if controlnet_enabled:
            generation_kwargs["image"] = _resize_condition_image(pose_image, width, height)
            generation_kwargs["controlnet_conditioning_scale"] = CONTROLNET_CONDITIONING_SCALE

        _log(
            f"[JOB] image generation started "
            f"width={width} height={height} steps={steps} "
            f"guidance_scale={guidance_scale} seed={seed} "
            f"identity_enabled={identity_enabled} "
            f"ip_adapter_ready={getattr(pipe, '_identity_adapter_ready', False)} "
            f"controlnet_enabled={controlnet_enabled}"
        )

        with torch.inference_mode():
            image, identity_used, identity_fallback_reason = _run_generation_with_identity_fallback(
                pipe,
                generation_kwargs,
                identity_enabled=identity_enabled and getattr(pipe, "_identity_adapter_ready", False),
            )

        buffer = io.BytesIO()

        fmt = "JPEG" if OUTPUT_FORMAT in {"jpg", "jpeg"} else "PNG"
        mime_type = "image/jpeg" if fmt == "JPEG" else "image/png"

        if fmt == "JPEG":
            image.save(buffer, format=fmt, quality=95)
        else:
            image.save(buffer, format=fmt)

        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        elapsed_ms = int((time.time() - started) * 1000)

        _log(f"[JOB] image generation completed elapsed_ms={elapsed_ms} seed={seed}")

        return {
            "image_base64": image_base64,
            "mime_type": mime_type,
            "format": OUTPUT_FORMAT,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "identity_enabled": identity_enabled,
            "identity_used": identity_used,
            "identity_fallback_reason": identity_fallback_reason,
            "ip_adapter_ready": getattr(pipe, "_identity_adapter_ready", False),
            "controlnet_enabled": controlnet_enabled,
            "controlnet_conditioning_scale": CONTROLNET_CONDITIONING_SCALE if controlnet_enabled else None,
            "model": MODEL_ID,
            "variant": MODEL_VARIANT,
            "elapsed_ms": elapsed_ms,
        }

    except Exception as exc:
        traceback.print_exc()

        return {
            "error": str(exc),
            "trace": traceback.format_exc()[-2500:],
            "model": MODEL_ID,
            "variant": MODEL_VARIANT,
            "elapsed_ms": int((time.time() - started) * 1000),
        }


_log("[BOOT] Worker process started. Waiting for RunPod jobs.")
runpod.serverless.start({"handler": handler})
