from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Tuple

from diffusers import AutoencoderKL  # fallback for older/nonstandard configs
from huggingface_hub import hf_hub_download


def _split_repo_and_subfolder(spec: str) -> Tuple[str, str | None]:
    """
    Accepts:
      - "org/repo"
      - "org/repo:subfolder"   (e.g. "black-forest-labs/FLUX.2-klein-4B:vae")
    If `spec` exists as a local path, we do NOT treat ':' specially.
    """
    if ":" in spec and not Path(spec).exists():
        repo, sub = spec.split(":", 1)
        sub = sub.strip("/") or None
        return repo, sub
    return spec, None


def _resolve_diffusers_class(class_name: str):
    """
    Best-effort resolver for diffusers classes named in config.json "_class_name".
    """
    import diffusers  # local import so module import errors are clearer

    # Many classes are exported at top-level (diffusers.AutoencoderKL, etc.)
    cls = getattr(diffusers, class_name, None)
    if cls is not None:
        return cls

    # Fallback: try common internal modules where model classes live
    for mod in (
        "diffusers.models",
        "diffusers.models.autoencoders",
        "diffusers.models.autoencoders.autoencoder_kl",
    ):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        cls = getattr(m, class_name, None)
        if cls is not None:
            return cls

    return None


def load_vae_auto(spec: str):
    """
    Load a diffusers VAE by inspecting its config.json to discover the correct class.

    `spec` can be:
      - Local path to a VAE directory containing config.json, OR
      - HF repo id, optionally with ":subfolder" (e.g. "org/repo:vae")

    Returns: an instantiated VAE model (already loaded, not moved to device).
    """
    p = Path(spec)

    if p.exists():
        cfg_path = p / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json at: {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        class_name = cfg.get("_class_name")
        vae_source = str(p)
        subfolder = None
    else:
        repo_id, subfolder = _split_repo_and_subfolder(spec)
        cfg_file = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            subfolder=subfolder,
        )
        cfg = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
        class_name = cfg.get("_class_name")
        vae_source = repo_id  # for from_pretrained()

    VaeCls = None
    if isinstance(class_name, str) and class_name:
        VaeCls = _resolve_diffusers_class(class_name)

    if VaeCls is None:
        # Conservative fallback: works for many VAEs, and fails loudly if incompatible.
        VaeCls = AutoencoderKL

    if subfolder is None:
        return VaeCls.from_pretrained(vae_source)

    return VaeCls.from_pretrained(vae_source, subfolder=subfolder)
