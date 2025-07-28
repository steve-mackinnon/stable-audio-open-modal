from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import modal
from generate_audio_sample import generate_audio_sample
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

auth_scheme = HTTPBearer()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1-dev")
    .pip_install(
        "fastapi[standard]==0.115.5",
        "pydantic==2.9.2",
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "stable-audio-tools==0.0.16",
        "sndfile==0.2.0",
    )
    .shell(["pip install -U flash-attn --no-build-isolation"])
    .add_local_file("src/generate_audio_sample.py", "/root/generate_audio_sample.py")
    .add_local_file("src/transient_detector.py", "/root/transient_detector.py")
)

app = modal.App("sample-gen")

volume = modal.Volume.from_name("models", create_if_missing=True)

MODEL_DIR = "/models/stable-audio-open"
MODEL_CONFIG_NAME = "model_config.json"
MODEL_CKPT_NAME = "model.safetensors"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/models": volume},
)
def download_model():
    """Download the model to the volume. Run this before deploying."""
    from huggingface_hub import hf_hub_download

    repo_id = "stabilityai/stable-audio-open-1.0"
    os.makedirs(MODEL_DIR, exist_ok=True)
    token = os.environ["HF_TOKEN"]
    hf_hub_download(
        repo_id,
        filename=MODEL_CONFIG_NAME,
        repo_type="model",
        local_dir=MODEL_DIR,
        token=token,
    )
    hf_hub_download(
        repo_id,
        filename=MODEL_CKPT_NAME,
        repo_type="model",
        token=token,
        local_dir=MODEL_DIR,
    )
    volume.commit()


@app.cls(
    gpu="any",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/models": volume},
)
class Model:
    pass


class GenerateSampleRequest(BaseModel):
    prompt: str
    cfg_scale: int = 10
    steps: int = 50
    trim_extra_hits: bool = True
    length: float = 0.7


@app.function(
    image=image,
    gpu="A10G",
    secrets=[
        modal.Secret.from_name("sample-gen-auth-token"),
    ],
    volumes={"/models": volume},
)
@modal.fastapi_endpoint(method="POST")
def generate_sample(
    body: GenerateSampleRequest,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    import json
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    import time

    volume.reload()
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    print("loading model")
    model_load_start = time.time()
    with open(f"{MODEL_DIR}/model_config.json") as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    model_ckpt_path = f"{MODEL_DIR}/model.safetensors"
    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))
    model_load_end = time.time()
    print(f"model loaded in {model_load_end - model_load_start} seconds")

    print(
        f"generating audio sample with prompt: {body.prompt} cfg_scale: {body.cfg_scale} steps: {body.steps}"
    )
    audio_gen_start = time.time()
    audio_bytes = generate_audio_sample(
        model=model,
        prompt=body.prompt,
        cfg_scale=body.cfg_scale or 10,
        steps=body.steps or 50,
        length=body.length or 0.7,
        trim_extra_hits=body.trim_extra_hits or True,
    )
    audio_gen_end = time.time()
    print(f"audio generated in {audio_gen_end - audio_gen_start} seconds")
    return StreamingResponse(audio_bytes, media_type="audio/wav")


@app.local_entrypoint()
def main():
    token = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials=os.environ["STABLE_AUDIO_MODAL_TOKEN"]
    )
    generate_sample.remote(
        GenerateSampleRequest(prompt="Beefy hip hop kick", cfg_scale=10, steps=50),
        token,
    )
