import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class CosmosCIEncoder(nn.Module):
    def __init__(
        self,
        model_id="nvidia/Cosmos-0.1-Tokenizer-CI8x8",
        image_size=None,
        torch_dtype="float16",
        patch_size=None,
        emb_dim=16,
    ):
        super().__init__()
        self.name = "cosmos_ci"
        self.model_id = model_id
        self.image_size = image_size
        self.torch_dtype = torch_dtype
        self.base_emb_dim = emb_dim
        self.emb_dim = emb_dim

        self._ensure_valid_config(model_id)
        if patch_size is None:
            if "CI8x8" in model_id:
                patch_size = 8
            elif "CI16x16" in model_id:
                patch_size = 16
        self.patch_size = patch_size

        self.base_model = self._load_tokenizer_model(
            model_id=model_id,
            torch_dtype=getattr(torch, torch_dtype),
        )
        self.base_model.eval()
        self.latent_ndim = 2

    def forward(self, x):
        if self.image_size is not None and x.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        if x.dtype != getattr(torch, self.torch_dtype):
            x = x.to(dtype=getattr(torch, self.torch_dtype))

        latents = self.base_model.encode(x)
        if latents.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W) latents, got shape {latents.shape}")

        b, c, h, w = latents.shape
        if c != self.emb_dim:
            raise ValueError(f"Latent channel mismatch: got {c}, expected {self.emb_dim}.")
        tokens = latents.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return tokens

    @staticmethod
    def _sanitize_json(text):
        return re.sub(r",(\s*[}\]])", r"\1", text)

    @classmethod
    def _ensure_valid_config(cls, model_id):
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()
        try:
            json.loads(raw)
            return
        except json.JSONDecodeError:
            sanitized = cls._sanitize_json(raw)
            json.loads(sanitized)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(sanitized)

    @staticmethod
    def _load_tokenizer_model(model_id, torch_dtype):
        class_name = "modeling_cosmos.CosmosTokenizer"
        try:
            model_cls = get_class_from_dynamic_module(
                class_reference=class_name,
                pretrained_model_name_or_path=model_id,
            )
            return model_cls.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            return AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
