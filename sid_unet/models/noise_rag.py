"""
Noise-RAG-Prototype-V1: Lightweight Forensic Model with Pretrained NoisePrint++ and Vector DB Retrieval.

Architecture Dataflow:
    Image (X)
      ├──> N = NoisePrint++(X) [Frozen Pretrained 17-layer DnCNN from TruFor]
      │      └──> NT = SomeConvVisionTransformerBlocks_N(N)
      │
      └──> XT = SomeConvVisionTransformerBlocks_X(X)
             │
             ├──> XR = Retrieve(XT) [Dense Retrieval from 100k MaskCLIP Pre-TVCA Vector DB]
             │
             ├──> XS = StopGrad(XR - XT) + XT  [Surrogate Residual Replacement]
             │
             ├──> Alignment Loss L_align(XT, StopGrad(XS)) [MSE / Entropy / KL]
             │
             └──> O_1 = XS
                  O_{i+1} = CrossAttention(Q=NT, K=O_i, V=O_i) + LightViTBlock
                  Output maps {O_1, ..., O_{L+1}} with multi-scale deep supervision.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.forensic_memory import stop_gradient_query_replacement

logger = logging.getLogger("sid_unet.models.noise_rag")

DEFAULT_NOISEPRINT_URL = (
    "https://raw.githubusercontent.com/grip-unina/TruFor/main/TruFor_train_test/"
    "pretrained_models/noiseprint++/noiseprint++.th"
)


class NoisePrintPlusPlus(nn.Module):
    """
    NoisePrint++ Forensic Feature Extractor (17-layer DnCNN from TruFor).

    Loads genuine pretrained weights online or from a local checkpoint with strict verification.
    Zero fallback to random/dummy weights: missing or mismatched weights raise an immediate RuntimeError.

    Args:
        pretrained (bool): Whether to load pretrained weights (default: True).
        weights_path_or_url (Optional[str]): Local path or URL to `noiseprint++.th`.
        freeze (bool): Whether to freeze all parameters (default: True).
        cache_dir (Optional[str]): Directory to cache downloaded weights.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights_path_or_url: Optional[str] = None,
        freeze: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.freeze = bool(freeze)
        self.weights_source = weights_path_or_url or DEFAULT_NOISEPRINT_URL

        # Build 17-layer DnCNN architecture matching TruFor noiseprint++
        num_levels = 17
        features = [64] * 16 + [1]
        bns = [False] + [True] * 15 + [False]
        acts = ["relu"] * 16 + ["linear"]
        in_c = 3

        layers: List[nn.Module] = []
        for i in range(num_levels):
            out_c = features[i]
            bias = not bns[i]
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=bias))
            if bns[i]:
                layers.append(nn.BatchNorm2d(out_c, momentum=0.1))
            if acts[i] == "relu":
                layers.append(nn.ReLU(inplace=True))
            in_c = out_c

        self.dncnn = nn.Sequential(*layers)

        if pretrained:
            self._load_pretrained(self.weights_source, cache_dir=cache_dir)
        else:
            raise ValueError(
                "NoisePrintPlusPlus requires pretrained=True. Fallback to random weights is strictly disabled."
            )

        if self.freeze:
            for p in self.dncnn.parameters():
                p.requires_grad = False
            self.eval()

    def _load_pretrained(self, source: str, cache_dir: Optional[str] = None) -> None:
        """Download or load genuine noiseprint++.th weights with strict state_dict verification."""
        try:
            if os.path.isfile(source):
                ckpt = torch.load(source, map_location="cpu", weights_only=False)
            else:
                ckpt = torch.hub.load_state_dict_from_url(
                    source,
                    model_dir=cache_dir,
                    map_location="cpu",
                    check_hash=False,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download/load NoisePrint++ weights from '{source}': {e}. "
                f"Strict zero-fallback policy: aborting initialization."
            ) from e

        if not isinstance(ckpt, dict) or "network" not in ckpt:
            raise ValueError(
                f"Invalid NoisePrint++ checkpoint payload from '{source}': expected dict with key 'network'."
            )

        net_sd = ckpt["network"]
        try:
            missing, unexpected = self.dncnn.load_state_dict(net_sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"NoisePrint++ state_dict mismatch: missing={missing}, unexpected={unexpected}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed strict load_state_dict for NoisePrint++: {e}. Zero-fallback policy enforced."
            ) from e

        logger.info(f"✅ Loaded genuine NoisePrint++ pretrained weights from: {source}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract NoisePrint++ residual.

        Args:
            x: RGB image tensor `[B, 3, H, W]` normalized in range `[0, 1]`.

        Returns:
            Noise residual tiled to 3 channels `[B, 3, H, W]`.
        """
        if self.freeze:
            with torch.no_grad():
                res = self.dncnn(x)
        else:
            res = self.dncnn(x)

        # Output is [B, 1, H, W]; tile to 3 channels to match standard vision backbone inputs (as in TruFor)
        if res.shape[1] == 1:
            res = torch.tile(res, (1, 3, 1, 1))
        return res


class LightTransformerBlock(nn.Module):
    """
    Lightweight Pre-LayerNorm Vision Transformer Block for spatial token maps [B, C, H, W].
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape `[B, C, H, W]`.

        Returns:
            Refined tensor of shape `[B, C, H, W]`.
        """
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]

        # Self-Attention with Pre-LN
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out

        # MLP with Pre-LN
        tokens = tokens + self.mlp(self.norm2(tokens))

        out = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out


class LightCrossAttentionBlock(nn.Module):
    """
    Lightweight Cross-Attention Block fusing Noise Tokens (Q = NT) with Visual/Prior Tokens (K, V = O_i).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_map: Noise tokens NT `[B, C, H, W]`.
            kv_map: State tokens O_i `[B, C, H, W]`.

        Returns:
            Next state O_{i+1} of shape `[B, C, H, W]`.
        """
        b, c, h, w = q_map.shape
        q_tokens = q_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        kv_tokens = kv_map.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Cross Attention: Q from NT, K and V from O_i
        q_norm = self.norm_q(q_tokens)
        kv_norm = self.norm_kv(kv_tokens)
        attn_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)

        # Residual connection onto kv_tokens (updating O_i with noise guidance)
        tokens = kv_tokens + attn_out

        # FFN
        tokens = tokens + self.mlp(self.norm_mlp(tokens))

        out = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out


class SomeConvVisionTransformerBlocks(nn.Module):
    """
    Hybrid Conv + Lightweight Vision Transformer Encoder.

    Downsamples input representation via a small pretrained conv backbone (e.g. timm mobilenetv3_small_100
    or resnet18 early stages) and refines spatial-contextual tokens using 2-4 lightweight ViT blocks.

    Args:
        conv_backbone (str): Name of timm pretrained model (default: "mobilenetv3_small_100").
        pretrained (bool): Whether to load online pretrained weights for conv stem (default: True).
        embed_dim (int): Output feature channels (default: 256).
        num_vit_blocks (int): Number of lightweight Transformer blocks (default: 2).
        out_stride (int): Downsampling stride (8 or 16, default: 8).
    """

    def __init__(
        self,
        conv_backbone: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        embed_dim: int = 256,
        num_vit_blocks: int = 2,
        out_stride: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_backbone_name = conv_backbone
        self.out_stride = int(out_stride)

        import timm

        # Determine feature index for chosen stride
        feat_idx = 2 if self.out_stride == 8 else 3
        try:
            self.conv_stem = timm.create_model(
                conv_backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(feat_idx,),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrained conv backbone '{conv_backbone}' via timm: {e}. "
                f"Zero fallback to uninitialized weights is permitted."
            ) from e

        stem_channels = self.conv_stem.feature_info.channels()[-1]

        # Project stem channels to embed_dim (256)
        self.proj = nn.Sequential(
            nn.Conv2d(stem_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # 2D Positional Embedding
        self.pos_embed: Optional[nn.Parameter] = None

        # Lightweight Transformer Blocks
        self.blocks = nn.ModuleList([
            LightTransformerBlock(embed_dim=embed_dim, num_heads=8, mlp_ratio=2.0)
            for _ in range(num_vit_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor `[B, 3, H, W]`.

        Returns:
            Token map `[B, embed_dim, H', W']`.
        """
        feats = self.conv_stem(x)[0]
        x_proj = self.proj(feats)

        b, c, h, w = x_proj.shape
        if self.pos_embed is None or self.pos_embed.shape[2:] != (h, w):
            # Dynamically initialize or interpolate 2D sinusoidal position embedding
            pe = self._build_2d_sincos_pos_embed(embed_dim=c, h=h, w=w, device=x.device, dtype=x_proj.dtype)
            self.pos_embed = nn.Parameter(pe, requires_grad=False)

        x_tokens = x_proj + self.pos_embed

        for blk in self.blocks:
            x_tokens = blk(x_tokens)

        return x_tokens

    @staticmethod
    def _build_2d_sincos_pos_embed(
        embed_dim: int, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Construct deterministic 2D sinusoidal position embeddings."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        half_dim = embed_dim // 4
        omega = torch.arange(half_dim, device=device, dtype=torch.float32) / half_dim
        omega = 1.0 / (10000 ** omega)

        out_y = grid_y.flatten()[:, None] * omega[None, :]
        out_x = grid_x.flatten()[:, None] * omega[None, :]

        pe = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
        if pe.shape[-1] < embed_dim:
            pe = F.pad(pe, (0, embed_dim - pe.shape[-1]))
        pe = pe.reshape(1, h, w, embed_dim).permute(0, 3, 1, 2).to(dtype=dtype)
        return pe


class ForensicVectorDBRetrieval(nn.Module):
    """
    GPU-accelerated dense nearest-neighbor retrieval over cached MaskCLIP pre-TVCA vector database.

    Maintains stored 256-D keys and continuous patch ground-truth tampered fraction labels in fp16/bf16.
    Fuses top-K retrieved vectors and patch labels to produce representation XR matching XT.

    Args:
        memory_path (Optional[str]): File path to `mask_clip_db_mae.pt`.
        memory_dtype (str): Storage precision ("float16" or "bfloat16").
        retrieval_k (int): Number of nearest neighbors to retrieve (default: 4).
        temperature (float): Softmax temperature for similarity weighting (default: 0.1).
        embed_dim (int): Vector representation dimension (default: 256).
    """

    def __init__(
        self,
        memory_path: Optional[str] = None,
        memory_dtype: str = "float16",
        retrieval_k: int = 4,
        temperature: float = 0.1,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.memory_path = memory_path
        self.memory_dtype_str = str(memory_dtype).lower().strip()
        self.storage_dtype = torch.bfloat16 if "bf16" in self.memory_dtype_str or "bfloat16" in self.memory_dtype_str else torch.float16
        self.retrieval_k = int(retrieval_k)
        self.temperature = float(temperature)
        self.embed_dim = int(embed_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Fuses retrieved 256-D vectors and 1-D patch labels back to embed_dim
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Database buffer registration
        self.register_buffer("db_keys", torch.empty((0, embed_dim), dtype=self.storage_dtype), persistent=False)
        self.register_buffer("db_labels", torch.empty((0, 1), dtype=self.storage_dtype), persistent=False)

        if memory_path:
            try:
                self.load_database(memory_path)
            except Exception as load_err:
                logger.warning(f"Could not load vector database from memory_path '{memory_path}': {load_err}")

    @property
    def has_database(self) -> bool:
        return self.db_keys.shape[0] > 0

    def load_database(self, filepath: str) -> None:
        """
        Load vector database from a standalone PyTorch file (.pt) or Hugging Face repository.
        Supports local file paths, candidate resolution paths, and Hugging Face Hub dataset repo IDs.
        """
        resolved_path = None

        # 1. Direct path check
        if os.path.isfile(filepath):
            resolved_path = os.path.abspath(filepath)

        # 2. Check if filepath is a Hugging Face Hub repository ID (or 'hf://<repo_id>')
        if resolved_path is None:
            clean_repo = filepath.replace("hf://", "").strip()
            if "/" in clean_repo and not clean_repo.startswith((".", "/")):
                repo_id = clean_repo
                target_filename = "mask_clip_db_mae.pt"
                parts = clean_repo.split("/")
                if len(parts) > 2 and parts[-1].endswith(".pt"):
                    repo_id = f"{parts[0]}/{parts[1]}"
                    target_filename = "/".join(parts[2:])
                try:
                    from huggingface_hub import hf_hub_download
                    resolved_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=target_filename,
                        repo_type="dataset",
                    )
                    logger.info(f"Downloaded vector database from Hugging Face Hub '{repo_id}': {resolved_path}")
                except Exception as hf_err:
                    logger.debug(f"HF Hub download check failed for '{filepath}': {hf_err}")

        # 3. Check candidate local directories if path doesn't exist directly
        if resolved_path is None:
            filename = os.path.basename(filepath) if filepath.endswith(".pt") else "mask_clip_db_mae.pt"
            candidates = [
                filepath,
                os.path.join("outputs", "dataset_cache", filename),
                os.path.join("outputs", "cache", "mask_clip_db_mae", filename),
                os.path.join("outputs", "dataset_cache", "mask_clip_db_mae.pt"),
                os.path.join("outputs", "cache", "mask_clip_db_mae", "mask_clip_db_mae.pt"),
                filename,
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    resolved_path = os.path.abspath(cand)
                    logger.info(f"Resolved vector database from candidate path: {resolved_path}")
                    break

        if resolved_path is None or not os.path.exists(resolved_path):
            raise FileNotFoundError(f"MaskCLIP vector database not found at '{filepath}' (and could not be resolved from local cache or HF Hub).")

        ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "keys" not in ckpt:
            raise ValueError(f"Invalid vector DB payload from {resolved_path}: missing 'keys'")

        keys = ckpt["keys"].to(dtype=self.storage_dtype)
        if "labels" in ckpt and ckpt["labels"] is not None:
            labels = ckpt["labels"].to(dtype=self.storage_dtype)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
        else:
            labels = torch.zeros((keys.shape[0], 1), dtype=self.storage_dtype)

        self.db_keys = keys
        self.db_labels = labels
        self.memory_path = resolved_path
        logger.info(
            f"✅ Loaded {keys.shape[0]} MaskCLIP pre-TVCA vectors into retrieval buffer "
            f"({keys.shape[1]} dim, {self.storage_dtype}) from: {resolved_path}"
        )

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Query vector database using visual tokens x_t.

        Args:
            x_t: Visual tokens `[B, C, H', W']`.

        Returns:
            Retrieved representation x_r `[B, C, H', W']`.
        """
        b, c, h, w = x_t.shape

        if not self.has_database:
            # Fallback path if memory database has not yet been extracted:
            # Identity projection so model can still execute forward passes for verification
            return x_t

        orig_dtype = x_t.dtype
        device = x_t.device
        k_entries = self.db_keys.to(device=device, dtype=self.storage_dtype)
        l_entries = self.db_labels.to(device=device, dtype=self.storage_dtype)

        tokens = x_t.permute(0, 2, 3, 1).reshape(-1, c)  # [B * H' * W', C]
        q_dtype = self.query_proj[0].weight.dtype
        q = self.query_proj(tokens.to(dtype=q_dtype)).to(dtype=self.storage_dtype)

        # L2-normalize for fast cosine similarity via matrix multiplication
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k_entries, p=2, dim=-1)

        k_actual = min(self.retrieval_k, k_entries.shape[0])
        chunk_size = 2048
        num_tokens = q_norm.shape[0]

        if num_tokens <= chunk_size:
            sim = torch.matmul(q_norm, k_norm.t())
            topk_sims, topk_indices = torch.topk(sim, k=k_actual, dim=-1, largest=True, sorted=True)
            attn_weights = F.softmax(topk_sims / max(self.temperature, 1e-4), dim=-1).unsqueeze(-1)
            ret_k = k_entries[topk_indices]
            ret_l = l_entries[topk_indices]
            agg_k = (attn_weights * ret_k).sum(dim=1)
            agg_l = (attn_weights * ret_l).sum(dim=1)
        else:
            agg_k_list = []
            agg_l_list = []
            for start_idx in range(0, num_tokens, chunk_size):
                end_idx = min(start_idx + chunk_size, num_tokens)
                q_chunk = q_norm[start_idx:end_idx]
                sim_chunk = torch.matmul(q_chunk, k_norm.t())
                topk_sims, topk_indices = torch.topk(sim_chunk, k=k_actual, dim=-1, largest=True, sorted=True)
                attn_weights = F.softmax(topk_sims / max(self.temperature, 1e-4), dim=-1).unsqueeze(-1)
                ret_k = k_entries[topk_indices]
                ret_l = l_entries[topk_indices]
                agg_k_list.append((attn_weights * ret_k).sum(dim=1))
                agg_l_list.append((attn_weights * ret_l).sum(dim=1))
            agg_k = torch.cat(agg_k_list, dim=0)
            agg_l = torch.cat(agg_l_list, dim=0)

        fuse_dtype = self.fuse_proj[0].weight.dtype
        fused_in = torch.cat([agg_k, agg_l], dim=-1).to(dtype=fuse_dtype)
        fused = self.fuse_proj(fused_in).to(dtype=orig_dtype)
        x_r = fused.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_r


class NoiseRAGPrototypeV1(nn.Module):
    """
    Noise-RAG-Prototype-V1 Forensic Model.

    Combines:
    1. Pretrained NoisePrint++ (TruFor 17-layer DnCNN)
    2. ConvStem + Lightweight ViT blocks for visual (XT) and noise (NT) representations
    3. Dense MaskCLIP pre-TVCA vector database retrieval (XR)
    4. Surrogate stop-gradient residual replacement (XS = StopGrad(XR - XT) + XT)
    5. XS-XT proximity / alignment loss (MSE / Entropy / KL)
    6. Iterative noise-guided cross-attention layers producing multi-layer outputs {O_1, ..., O_{L+1}}
    7. Multi-scale deep supervision compatible with SIDTotalLoss.

    Args:
        noiseprint_weights (Optional[str]): Path or URL to pretrained `noiseprint++.th`.
        freeze_noiseprint (bool): Whether to freeze NoisePrint++ (default: True).
        memory_path (Optional[str]): Path to cached `mask_clip_db_mae.pt`.
        memory_dtype (str): Storage precision ("float16" or "bfloat16").
        embed_dim (int): Feature channel dimension (default: 256).
        conv_backbone (str): Lightweight pretrained conv stem (default: "mobilenetv3_small_100").
        num_vit_blocks (int): ViT blocks in visual and noise stems (default: 2).
        num_cross_stages (int): Number of recurrent cross-attention stages L (default: 3).
        out_stride (int): Spatial downsampling stride (default: 8).
        retrieval_k (int): Top-K retrieval count (default: 4).
        align_loss_type (str): "mse", "entropy", or "mse+entropy" (default: "mse").
        align_loss_weight (float): Multiplier for alignment loss (default: 0.2).
        entropy_temperature (float): Temperature for channel-wise entropy loss (default: 1.0).
        aux_classifier (bool): Whether to enable auxiliary class classification (default: True).
        num_classes (int): Number of target classes for classification (default: 2).
        in_channels (int): Input image channels (default: 3).
        out_channels (int): Output mask channels (default: 1).
    """

    def __init__(
        self,
        noiseprint_weights: Optional[str] = None,
        freeze_noiseprint: bool = True,
        memory_path: Optional[str] = None,
        memory_dtype: str = "float16",
        embed_dim: int = 256,
        conv_backbone: str = "mobilenetv3_small_100",
        num_vit_blocks: int = 2,
        num_cross_stages: int = 3,
        out_stride: int = 8,
        retrieval_k: int = 4,
        align_loss_type: str = "mse",
        align_loss_weight: float = 0.2,
        entropy_temperature: float = 1.0,
        aux_classifier: bool = True,
        num_classes: int = 2,
        in_channels: int = 3,
        out_channels: int = 1,
        **kwargs: Any,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_cross_stages = int(num_cross_stages)
        self.align_loss_type = str(align_loss_type).lower().strip()
        self.align_loss_weight = float(align_loss_weight)
        self.entropy_temperature = float(entropy_temperature)
        self.aux_classifier = bool(aux_classifier)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.memory_path = memory_path
        self.cross_mode = str(kwargs.get("cross_mode", "recurrent")).lower().strip()

        # 1. NoisePrint++ Extractor
        self.noiseprint = NoisePrintPlusPlus(
            pretrained=True,
            weights_path_or_url=noiseprint_weights,
            freeze=freeze_noiseprint,
        )

        # 2. Visual & Noise Hybrid Conv-ViT Encoders
        self.conv_vit_x = SomeConvVisionTransformerBlocks(
            conv_backbone=conv_backbone,
            pretrained=True,
            embed_dim=embed_dim,
            num_vit_blocks=num_vit_blocks,
            out_stride=out_stride,
        )
        self.conv_vit_n = SomeConvVisionTransformerBlocks(
            conv_backbone=conv_backbone,
            pretrained=True,
            embed_dim=embed_dim,
            num_vit_blocks=num_vit_blocks,
            out_stride=out_stride,
        )

        # 3. Dense MaskCLIP Vector Database Retrieval
        self.retrieval = ForensicVectorDBRetrieval(
            memory_path=memory_path,
            memory_dtype=memory_dtype,
            retrieval_k=retrieval_k,
            embed_dim=embed_dim,
        )

        # 4. Iterative Cross-Attention Stages: Q = NT, K = O_i, V = O_i
        self.cross_stages = nn.ModuleList([
            LightCrossAttentionBlock(embed_dim=embed_dim, num_heads=8, mlp_ratio=2.0)
            for _ in range(self.num_cross_stages)
        ])

        # 5. Layer Prediction Heads for Deep Supervision {O_1, ..., O_{L+1}}
        # Total output maps = num_cross_stages + 1
        num_layers = self.num_cross_stages + 1
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, out_channels, kernel_size=1),
            )
            for _ in range(num_layers)
        ])

        # Final Fused Mask Head
        self.final_head = nn.Sequential(
            nn.Conv2d(embed_dim * num_layers, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

        # Auxiliary Classification Head
        if self.aux_classifier:
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        else:
            self.class_head = None

    def compute_alignment_loss(self, x_t: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        """
        Compute proximity / alignment loss pulling visual tokens XT close to StopGrad(XS) (StopGrad(XR)).

        Args:
            x_t: Visual encoder tokens `[B, C, H', W']`.
            x_s: Surrogate replaced representation `[B, C, H', W']`.

        Returns:
            Scalar alignment loss tensor.
        """
        target_s = x_s.detach()

        mse_loss = F.mse_loss(x_t, target_s)
        if self.align_loss_type == "mse":
            return mse_loss * self.align_loss_weight

        # Channel-wise temperature-scaled Cross-Entropy / KL Divergence
        tau = max(self.entropy_temperature, 1e-4)
        p_s = F.softmax(target_s / tau, dim=1)
        log_p_t = F.log_softmax(x_t / tau, dim=1)
        entropy_loss = F.kl_div(log_p_t, p_s, reduction="batchmean")

        if self.align_loss_type == "entropy":
            return entropy_loss * self.align_loss_weight
        elif self.align_loss_type in ("mse+entropy", "entropy+mse"):
            return (mse_loss + entropy_loss) * self.align_loss_weight
        else:
            return mse_loss * self.align_loss_weight

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        End-to-End Forward Pass of Noise-RAG-Prototype-V1.

        Args:
            x: Input RGB image tensor `[B, 3, H, W]` in range `[0, 1]`.
            return_dict: If True, returns rich dictionary with multiscale logits and alignment loss.

        Returns:
            Dictionary with keys:
                - `mask_logits`: Final prediction mask `[B, 1, H, W]`.
                - `multiscale_logits`: List of layer-specific prediction masks `[M_1, ..., M_{L+1}]`.
                - `artifact_loss`: Alignment loss between XT and XS (for SIDTotalLoss).
                - `align_loss`: Raw alignment loss value.
                - `pred_mask`: Sigmoid probabilities `[B, 1, H, W]`.
                - `class_logits`: Class prediction logits `[B, num_classes]` (if aux_classifier).
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 1. NoisePrint++ Extraction
        n_res = self.noiseprint(x)  # [B, 3, H, W]

        # 2. Visual and Noise Encoders
        x_t = self.conv_vit_x(x)      # [B, embed_dim, H', W']
        n_t = self.conv_vit_n(n_res)  # [B, embed_dim, H', W']

        # 3. Dense Retrieval from MaskCLIP Vector DB
        x_r = self.retrieval(x_t)

        # 4. Surrogate Stop-Gradient Replacement
        # Forward: x_s = x_r; Backward: d(x_s)/d(x_t) = I, d(x_s)/d(x_r) = 0
        x_s = stop_gradient_query_replacement(original_query=x_t, retrieved_query=x_r)

        # 5. Proximity / Alignment Loss
        align_loss = self.compute_alignment_loss(x_t=x_t, x_s=x_s)

        # 6. Iterative Noise-Guided Cross-Attention Layers
        layer_maps: List[torch.Tensor] = [x_s]
        curr_state = x_s
        for stage in self.cross_stages:
            kv_input = x_s if self.cross_mode in ("xs", "surrogate") else curr_state
            curr_state = stage(q_map=n_t, kv_map=kv_input)
            layer_maps.append(curr_state)

        # 7. Layer Prediction Heads with Bilinear Upsampling to full resolution
        multiscale_logits: List[torch.Tensor] = []
        for i, (feat, head) in enumerate(zip(layer_maps, self.layer_heads)):
            logits_down = head(feat)
            logits_full = F.interpolate(logits_down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            multiscale_logits.append(logits_full)

        # Fused Final Prediction Mask
        concat_feats = torch.cat(layer_maps, dim=1)  # [B, embed_dim * (L+1), H', W']
        final_down = self.final_head(concat_feats)
        final_logits = F.interpolate(final_down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        # Optional Auxiliary Classifier
        class_logits = self.class_head(layer_maps[-1]) if self.class_head is not None else None

        if return_dict:
            res: Dict[str, Any] = {
                "mask_logits": final_logits,
                "multiscale_logits": multiscale_logits,
                "artifact_loss": align_loss,
                "align_loss": align_loss,
                "pred_mask": torch.sigmoid(final_logits),
                "features": layer_maps[-1],
            }
            if class_logits is not None:
                res["class_logits"] = class_logits
                res["pred_label"] = torch.argmax(class_logits, dim=1)
            return res

        if self.aux_classifier and class_logits is not None:
            return final_logits, class_logits
        return final_logits
