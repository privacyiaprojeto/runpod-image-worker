import base64
import io
import os
import time
import traceback
from typing import Any, Dict

import runpod
import torch
from diffusers import AutoPipelineForText2Image, EulerAncestralDiscreteScheduler

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

DEFAULT_NEGATIVE_PROMPT = os.getenv(
    "DEFAULT_NEGATIVE_PROMPT",
    "low quality, blurry, distorted face, bad anatomy, extra fingers, missing fingers, deformed hands, watermark, text, logo, duplicate person, oversaturated"
)

PIPE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _log(message: str) -> None:
    print(message, flush=True)


def _sanitize_prompt(value: Any, max_len: int = 1600) -> str:
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


def load_pipeline():
    global PIPE

    if PIPE is not None:
        return PIPE

    start = time.time()

    _log(
        f"[BOOT] Loading image model={MODEL_ID} "
        f"variant={MODEL_VARIANT} device={DEVICE}"
    )

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        variant=MODEL_VARIANT,
        use_safetensors=True,
    )

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

    PIPE = pipe

    _log(f"[BOOT] Image model ready in {round(time.time() - start, 2)}s")
    return PIPE


load_pipeline()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    payload = event.get("input") or {}

    prompt = _sanitize_prompt(payload.get("prompt"))
    if not prompt:
        return {"error": "prompt obrigatório"}

    negative_prompt = _sanitize_prompt(
        payload.get("negative_prompt") or payload.get("negativePrompt") or DEFAULT_NEGATIVE_PROMPT,
        max_len=1200,
    )

    width = _clamp_dimension(payload.get("width"), DEFAULT_WIDTH, MAX_WIDTH)
    height = _clamp_dimension(payload.get("height"), DEFAULT_HEIGHT, MAX_HEIGHT)
    steps = _clamp_int(payload.get("steps"), DEFAULT_STEPS, 8, 50)
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

    try:
        pipe = load_pipeline()

        generator = (
            torch.Generator(device=DEVICE).manual_seed(seed)
            if DEVICE == "cuda"
            else torch.Generator().manual_seed(seed)
        )

        _log(
            f"[JOB] image generation started "
            f"width={width} height={height} steps={steps} "
            f"guidance_scale={guidance_scale} seed={seed}"
        )

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

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


runpod.serverless.start({"handler": handler})