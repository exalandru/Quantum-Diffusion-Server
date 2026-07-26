"""
Petit serveur local exposant mflux via une API compatible OpenAI Images.
Route vers le bon binaire CLI mflux selon le modèle choisi (chaque famille
de modèle a sa propre commande : mflux-generate, mflux-generate-qwen,
mflux-generate-z-image-turbo, etc.)

Installation :
    uv tool install --upgrade mflux
    uv run --with fastapi --with uvicorn --with python-multipart python main.py

Le serveur écoute sur http://127.0.0.1:8765
Endpoints :
    POST /v1/images/generations   (texte -> image)
    POST /v1/images/edits         (image + texte -> image, si le modèle le supporte)

Aucun fichier n'est conservé sur disque : chaque image générée transite par
un fichier temporaire, lu puis supprimé avant que la réponse HTTP ne parte.

Les logs mflux (téléchargement, étapes de génération) sont affichés en temps
réel dans la console et écrits dans mflux.log.
"""

import asyncio
import base64
import json
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "mflux.log"

logger = logging.getLogger("mflux")
logger.setLevel(logging.INFO)

# Console handler — affichage en temps réel
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console_handler)

# File handler — persistance dans mflux.log
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)


CONFIG_PATH = Path(__file__).parent / "server.config"

SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="mflux_scratch_"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Serveur démarré — scratch dir : {SCRATCH_DIR}")
    yield
    # Cleanup au shutdown
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    logger.info("Serveur arrêté — scratch dir nettoyé")


app = FastAPI(title="mflux OpenAI-compatible bridge", lifespan=lifespan)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de config introuvable : {CONFIG_PATH}. "
            f"Crée-le à côté de ce script (voir mflux_config.json fourni)."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()
MODELS = CONFIG["models"]  # { "flux2-dev": {...}, "qwen-image": {...}, "z-image-turbo": {...} }
DEFAULT_MODEL_KEY = CONFIG["default_model"]


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = None          # "flux2-dev" | "qwen-image" | "z-image-turbo" ; sinon default_model
    n: int = 1
    size: str = None           # "WxH" ; sinon la taille par défaut du modèle choisi
    steps: int = None          # sinon les steps par défaut du modèle choisi
    response_format: str = "b64_json"  # "b64_json" ou "raw" (octets image/png bruts)
    seed: int = None           # seed pour la reproductibilité (optionnel)
    negative_prompt: str = None  # prompt négatif pour guider ce qu'on ne veut pas (optionnel)


def get_model_config(key: str | None):
    """Retourne (clé résolue, config du modèle) depuis mflux_config.json."""
    key = key or DEFAULT_MODEL_KEY
    if key not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu : '{key}'. Choix possibles : {list(MODELS.keys())}",
        )
    return key, MODELS[key]


async def run_mflux(model_key: str, prompt: str, steps: int, width: int, height: int,
                    image_path: Path = None, strength: float = None,
                    seed: int = None, negative_prompt: str = None) -> bytes:
    """
    Lance le binaire CLI mflux propre au modèle choisi, écrit sur un fichier
    temporaire éphémère, lit les octets produits, puis supprime le fichier.

    Stream stderr ligne par ligne pour afficher les logs en temps réel
    (téléchargement de modèle, progression de la génération, etc.).
    """
    _, mconf = get_model_config(model_key)
    out_file = SCRATCH_DIR / f"{uuid.uuid4().hex}.png"

    cmd = [
        mconf["cli"],  # ex: "mflux-generate", "mflux-generate-qwen", "mflux-generate-z-image-turbo"
        "--model", mconf["repo"],
        "--prompt", prompt,
        "--steps", str(steps),
        "--width", str(width),
        "--height", str(height),
        "--output", str(out_file),
    ]
    if mconf.get("quantize"):
        cmd += ["-q", str(mconf["quantize"])]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if negative_prompt is not None:
        cmd += ["--negative-prompt", negative_prompt]
    if image_path is not None:
        if not mconf.get("supports_image_to_image", False):
            raise HTTPException(
                status_code=400,
                detail=f"Le modèle '{model_key}' ne supporte pas l'image-to-image dans cette config.",
            )
        cmd += ["--image-path", str(image_path), "--image-strength", str(strength)]

    logger.info(f"▶ {mconf['cli']} — modèle={mconf['repo']}, taille={width}x{height}, steps={steps}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Stream stderr ligne par ligne en temps réel
    async def read_stderr():
        stderr_lines = []
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            stderr_lines.append(text)
            if text:
                logger.info(f"  {text}")
        return "\n".join(stderr_lines)

    # Lancer la lecture de stderr en parallèle du wait
    stderr_task = asyncio.create_task(read_stderr())
    await proc.wait()
    stderr_output = await stderr_task

    if proc.returncode != 0 or not out_file.exists():
        logger.error(f"✗ {mconf['cli']} a échoué (code={proc.returncode})")
        raise HTTPException(
            status_code=500,
            detail=f"{mconf['cli']} a échoué :\n{stderr_output[-2000:]}",
        )

    logger.info(f"✓ Image générée : {out_file.name}")

    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, out_file.read_bytes
        )
    finally:
        out_file.unlink(missing_ok=True)


@app.post("/v1/images/generations")
async def generate_image(req: ImageGenerationRequest):
    model_key, mconf = get_model_config(req.model)

    size_str = req.size or mconf["default_size"]
    try:
        width, height = (int(x) for x in size_str.lower().split("x"))
    except Exception:
        raise HTTPException(status_code=400, detail="size doit être au format 'WxH', ex: 1024x1024")

    steps = req.steps or mconf["default_steps"]

    if req.n == 1 and req.response_format == "raw":
        data = await run_mflux(model_key, req.prompt, steps, width, height,
                               seed=req.seed, negative_prompt=req.negative_prompt)
        return Response(content=data, media_type="image/png")

    results = []
    for i in range(max(1, req.n)):
        # Si un seed est fourni, on l'incrémente pour chaque image supplémentaire
        img_seed = req.seed + i if req.seed is not None else None
        data = await run_mflux(model_key, req.prompt, steps, width, height,
                               seed=img_seed, negative_prompt=req.negative_prompt)
        results.append({"b64_json": base64.b64encode(data).decode("utf-8")})

    return JSONResponse({"created": int(time.time()), "data": results})


@app.post("/v1/images/edits")
async def edit_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form(None),
    strength: float = Form(0.6),
    steps: int = Form(None),
    size: str = Form(None),
    response_format: str = Form("b64_json"),
    seed: int = Form(None),
    negative_prompt: str = Form(None),
):
    model_key, mconf = get_model_config(model)

    size_str = size or mconf["default_size"]
    try:
        width, height = (int(x) for x in size_str.lower().split("x"))
    except Exception:
        raise HTTPException(status_code=400, detail="size doit être au format 'WxH'")

    steps_val = steps or mconf["default_steps"]

    in_file = SCRATCH_DIR / f"in_{uuid.uuid4().hex}.png"
    in_file.write_bytes(await image.read())

    try:
        data = await run_mflux(
            model_key, prompt, steps_val, width, height,
            image_path=in_file, strength=strength,
            seed=seed, negative_prompt=negative_prompt,
        )
    finally:
        in_file.unlink(missing_ok=True)

    if response_format == "raw":
        return Response(content=data, media_type="image/png")

    return JSONResponse({"created": int(time.time()), "data": [{"b64_json": base64.b64encode(data).decode("utf-8")}]})


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": k, "object": "model"} for k in MODELS.keys()]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
