# Worker V2 — Identity / Control-ready

## Objetivo
Evolui o worker de text-to-image puro para geração com identidade visual opcional via IP-Adapter e suporte opcional a ControlNet quando uma imagem de pose/composição for enviada.

## ENV recomendadas no RunPod

```env
USE_IP_ADAPTER=true
IP_ADAPTER_STRICT=false
IP_ADAPTER_SCALE=0.65
IP_ADAPTER_REPO=h94/IP-Adapter
IP_ADAPTER_SUBFOLDER=sdxl_models
IP_ADAPTER_WEIGHT_NAME=ip-adapter-plus_sdxl_vit-h.safetensors
USE_CONTROLNET_OPENPOSE=false
CONTROLNET_STRICT=false
CONTROLNET_CONDITIONING_SCALE=0.8
DEFAULT_STEPS=30
DEFAULT_GUIDANCE_SCALE=7.0
```

## Payload compatível
O backend atual já envia `companion.avatar_url`; o worker usa esse campo como referência de identidade quando `USE_IP_ADAPTER=true`.

Campos opcionais aceitos:
- `identity_image_url`
- `identity_image_base64`
- `reference_image_url`
- `reference_image_base64`
- `pose_image_url`
- `pose_image_base64`

## Notas
- IP-Adapter melhora consistência visual, mas não garante 100% de identidade facial.
- ControlNet só ativa quando `USE_CONTROLNET_OPENPOSE=true` e uma imagem de pose/composição é enviada.
- Para FaceID real no futuro, será necessário avaliar insightface/onnxruntime e custo de VRAM.
