"""
Build and push a curated human-face domain-evaluation dataset to Hugging Face Hub.

Dataset: KhangTruong/face-forgery-domain-eval
Splits:
  - model1_in_domain : FaceForensics++ (C23 test pairs) — Model 1 (MSCCNet) in-domain faces + masks
  - model2_in_domain : Disjoint FaceForensics++ (C23 test pairs) — Model 2 (OmniFD FF++) in-domain faces + masks
  - out_of_domain    : Random human faces (nielsr/CelebA-faces) + cross-domain Celeb-DF-v2 — out-of-domain for both

Usage:
    python scripts/build_face_domain_dataset.py \
        --repo-id KhangTruong/face-forgery-domain-eval \
        --save-local outputs/face_domain_dataset
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

FF_ZIP_URL = "https://huggingface.co/datasets/Ella4839/DeepfakeBench/resolve/main/FaceForensics++.zip"
CELEBDF_ZIP_URL = "https://huggingface.co/datasets/Ella4839/DeepfakeBench/resolve/main/Celeb-DF-v2.zip"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only HTTP Range stream so zipfile reads only headers + selected entries."""

    def __init__(self, url: str, timeout: int = 30) -> None:
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray) -> int:
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.size - 1, self.pos + n - 1)
        resp = self.session.get(
            self.url,
            headers={"Range": f"bytes={self.pos}-{end}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.content
        count = len(data)
        b[:count] = data
        self.pos += count
        return count


def _read_zip_image(zf: zipfile.ZipFile, member: str, mode: str = "RGB") -> Image.Image:
    with zf.open(member, "r") as fp:
        return Image.open(io.BytesIO(fp.read())).convert(mode)


def _binarize_mask(mask_img: Image.Image) -> Image.Image:
    arr = np.asarray(mask_img.convert("L"), dtype=np.uint8)
    binary = ((arr >= 128).astype(np.uint8)) * 255
    return Image.fromarray(binary, mode="L")


def _zero_mask(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    return Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L")


def _compute_ssim_diff_mask(fake_img: Image.Image, real_img: Image.Image) -> Tuple[Image.Image, float]:
    """Compute binary facial manipulation mask via SSIM + convex hull (MSCCNet annotation_ff recipe)."""
    fake_np = np.asarray(fake_img.convert("RGB"), dtype=np.uint8)
    real_np = np.asarray(real_img.convert("RGB").resize(fake_img.size, Image.BILINEAR), dtype=np.uint8)
    _, diff = compare_ssim(real_np, fake_np, channel_axis=-1, full=True)
    diff_gray = np.mean(1.0 - diff, axis=2)
    mask = (diff_gray >= 0.12).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 64:
            hull = cv2.convexHull(cnt)
            cv2.fillPoly(hull_mask, [hull], 255)
    hull_mask = cv2.erode(hull_mask, np.ones((3, 3), np.uint8), iterations=1)
    area = float((hull_mask > 0).mean())
    return Image.fromarray(hull_mask, mode="L"), area


def _collect_ffpp_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract disjoint FaceForensics++ C23 test samples for Model 1 (MSCCNet) and Model 2 (OmniFD)."""
    stream = io.BufferedReader(HTTPRangeFile(FF_ZIP_URL), buffer_size=8 * 1024 * 1024)
    model1_rows: List[Dict[str, Any]] = []
    model2_rows: List[Dict[str, Any]] = []

    with zipfile.ZipFile(stream) as zf:
        test_pairs: List[List[str]] = json.loads(zf.read("test.json"))
        real_frames: Dict[str, List[str]] = defaultdict(list)
        fake_frames: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        fake_masks: Dict[Tuple[str, str, str], str] = {}

        for name in zf.namelist():
            if not name.endswith(".png"):
                continue
            if name.startswith("original_sequences/youtube/c23/frames/"):
                parts = name.split("/")
                if len(parts) == 6:
                    real_frames[parts[4]].append(name)
            elif name.startswith("manipulated_sequences/"):
                parts = name.split("/")
                if len(parts) == 6:
                    method, _, kind, pair_id, fname = parts[1], parts[2], parts[3], parts[4], parts[5]
                    if method not in {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}:
                        continue
                    if kind == "frames":
                        fake_frames[(method, pair_id)].append(name)
                    elif kind == "masks":
                        fake_masks[(method, pair_id, fname)] = name

        def _pick_split(pair_slice: List[List[str]], domain_role: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            # 4 real human faces from distinct test videos
            real_count = 0
            for vid_a, _ in pair_slice:
                cand_list = sorted(real_frames.get(vid_a, []))
                if not cand_list:
                    continue
                chosen = cand_list[len(cand_list) // 3]
                img = _read_zip_image(zf, chosen, "RGB")
                rows.append(
                    {
                        "image": img,
                        "mask": _zero_mask(img.size),
                        "label": 0,
                        "label_name": "authentic",
                        "domain_role": domain_role,
                        "source_dataset": "FaceForensics++_c23_real",
                        "manipulation_type": "real_youtube",
                        "image_id": f"{domain_role}_ff_real_{vid_a}_{os.path.splitext(os.path.basename(chosen))[0]}",
                    }
                )
                real_count += 1
                if real_count >= 4:
                    break

            # 8 manipulated faces (2 per FF++ method) with official masks
            methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            method_offset = 4
            for m_idx, method in enumerate(methods):
                added = 0
                for vid_a, vid_b in pair_slice[method_offset + m_idx * 2 :]:
                    pair_id = f"{vid_a}_{vid_b}"
                    cands = sorted(fake_frames.get((method, pair_id), []))
                    if not cands:
                        continue
                    # Pick a middle frame that has a non-empty official mask
                    for cand in [cands[len(cands) // 2], cands[len(cands) // 3], cands[0]]:
                        fname = os.path.basename(cand)
                        mask_member = fake_masks.get((method, pair_id, fname))
                        if not mask_member:
                            continue
                        mask_img = _binarize_mask(_read_zip_image(zf, mask_member, "L"))
                        if float((np.asarray(mask_img) > 0).mean()) < 0.005:
                            continue
                        img = _read_zip_image(zf, cand, "RGB")
                        if mask_img.size != img.size:
                            mask_img = mask_img.resize(img.size, Image.NEAREST)
                        rows.append(
                            {
                                "image": img,
                                "mask": mask_img,
                                "label": 1,
                                "label_name": "tampered",
                                "domain_role": domain_role,
                                "source_dataset": f"FaceForensics++_c23_{method}",
                                "manipulation_type": method,
                                "image_id": f"{domain_role}_ff_{method.lower()}_{pair_id}_{os.path.splitext(fname)[0]}",
                            }
                        )
                        added += 1
                        break
                    if added >= 2:
                        break
            return rows

        model1_rows = _pick_split(test_pairs[:35], "model1_in_domain_msccnet")
        model2_rows = _pick_split(test_pairs[35:70], "model2_in_domain_omnifd")

    return model1_rows, model2_rows


def _collect_ood_split() -> List[Dict[str, Any]]:
    """Collect out-of-domain human faces from nielsr/CelebA-faces and Celeb-DF-v2."""
    from datasets import load_dataset

    rows: List[Dict[str, Any]] = []

    # 1. Random/wild real faces from nielsr/CelebA-faces (6 diverse portraits)
    celeba = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
    picked_idx = {0, 35, 70, 105, 140, 175}
    for idx, ex in enumerate(celeba):
        if idx in picked_idx:
            img = ex["image"].convert("RGB")
            rows.append(
                {
                    "image": img,
                    "mask": _zero_mask(img.size),
                    "label": 0,
                    "label_name": "authentic",
                    "domain_role": "out_of_domain_both",
                    "source_dataset": "CelebA-faces",
                    "manipulation_type": "real_portrait",
                    "image_id": f"ood_celeba_{idx:04d}",
                }
            )
        if idx >= max(picked_idx):
            break

    # 2. Cross-domain Celebrity Deepfakes from Ella4839/DeepfakeBench/Celeb-DF-v2.zip
    stream = io.BufferedReader(HTTPRangeFile(CELEBDF_ZIP_URL), buffer_size=4 * 1024 * 1024)
    with zipfile.ZipFile(stream) as zf:
        real_by_clip: Dict[str, List[str]] = defaultdict(list)
        synth_by_pair: Dict[str, List[str]] = defaultdict(list)
        for name in zf.namelist():
            if not name.endswith(".png"):
                continue
            if name.startswith("Celeb-real/frames/"):
                parts = name.split("/")
                if len(parts) == 4:
                    real_by_clip[parts[2]].append(name)
            elif name.startswith("Celeb-synthesis/frames/"):
                parts = name.split("/")
                if len(parts) == 4:
                    synth_by_pair[parts[2]].append(name)

        # 2 real Celeb-DF-v2 faces from distinct identities
        for clip_id in ["id0_0000", "id1_0000"]:
            frames = sorted(real_by_clip.get(clip_id, []))
            if not frames:
                continue
            chosen = frames[len(frames) // 2]
            img = _read_zip_image(zf, chosen, "RGB")
            rows.append(
                {
                    "image": img,
                    "mask": _zero_mask(img.size),
                    "label": 0,
                    "label_name": "authentic",
                    "domain_role": "out_of_domain_both",
                    "source_dataset": "Celeb-DF-v2_real",
                    "manipulation_type": "real_celebrity_frame",
                    "image_id": f"ood_celebdf_real_{clip_id}_{os.path.splitext(os.path.basename(chosen))[0]}",
                }
            )

        # 4 manipulated Celeb-DF-v2 faces with SSIM diff mask against target real frame
        picked_synth = 0
        seen_targets = set()
        for pair_id in sorted(synth_by_pair.keys()):
            # Format: idX_idY_ZZZZ -> target real clip is idX_ZZZZ
            toks = pair_id.split("_")
            if len(toks) != 3:
                continue
            target_clip = f"{toks[0]}_{toks[2]}"
            if toks[0] in seen_targets or target_clip not in real_by_clip:
                continue
            real_map = {os.path.basename(p): p for p in real_by_clip[target_clip]}
            for synth_path in sorted(synth_by_pair[pair_id])[1:]:
                fname = os.path.basename(synth_path)
                if fname not in real_map:
                    continue
                fake_img = _read_zip_image(zf, synth_path, "RGB")
                real_img = _read_zip_image(zf, real_map[fname], "RGB")
                mask_img, area = _compute_ssim_diff_mask(fake_img, real_img)
                if 0.01 <= area <= 0.65:
                    rows.append(
                        {
                            "image": fake_img,
                            "mask": mask_img,
                            "label": 1,
                            "label_name": "tampered",
                            "domain_role": "out_of_domain_both",
                            "source_dataset": "Celeb-DF-v2_synthesis",
                            "manipulation_type": "CelebDF_faceswap",
                            "image_id": f"ood_celebdf_synth_{pair_id}_{os.path.splitext(fname)[0]}",
                        }
                    )
                    seen_targets.add(toks[0])
                    picked_synth += 1
                    break
            if picked_synth >= 4:
                break

    return rows


DATASET_CARD_MD = """---
license: mit
task_categories:
  - image-classification
  - image-segmentation
tags:
  - face-forgery-detection
  - deepfake-localization
  - msccnet
  - omnifd
  - out-of-domain
pretty_name: Face Forgery Domain Evaluation (MSCCNet & OmniFD)
---

# Face Forgery Domain Evaluation (`face-forgery-domain-eval`)

Curated human-face forgery detection and pixel-level localization evaluation dataset organized by domain alignment for **MSCCNet** (Model 1) and **OmniFD** (Model 2).

## Splits

| Split | Samples | Authentic / Tampered | Origin | Domain Role |
|---|---|---|---|---|
| `model1_in_domain` | 12 | 4 real / 8 fake | FaceForensics++ C23 (`test.json` pairs 0..34) | Model 1 (MSCCNet) In-Domain + Model 2 (OmniFD FF++) aligned |
| `model2_in_domain` | 12 | 4 real / 8 fake | FaceForensics++ C23 (`test.json` pairs 35..69, disjoint) | Model 2 (OmniFD `ff_image` / `ff_spatial`) In-Domain |
| `out_of_domain` | 12 | 8 real / 4 fake | `nielsr/CelebA-faces` (6) + `Celeb-DF-v2` (6) | Out-of-Domain human faces for both models |

## Source Provenance & Unavailable Datasets Note

- **FaceForensics++ C23** & **Celeb-DF-v2** frames/masks were extracted from `Ella4839/DeepfakeBench`. Official FF++ binary masks are used directly for manipulated samples in `model1_in_domain` and `model2_in_domain`; `Celeb-DF-v2` masks are computed via SSIM structural diff + morphological hull between target real and synthesized frames.
- **Random In-the-Wild Faces** were sampled from `nielsr/CelebA-faces` (`train` split) with all-zero ground-truth masks.
- *Upstream unavailability note*: ForgeryNet `public_test_images.tar` on Google Drive currently hits download quota limits, and the upstream FMLD SharePoint link returns HTTP 404. Because OmniFD officially releases pretrained checkpoints for FaceForensics++ (`omnifd_swin-s_20k_ff_image.pth`, `omnifd_swin-s_20k_ff_spatial.pth`) alongside ForgeryNet and FMLD, `model2_in_domain` uses disjoint FaceForensics++ C23 test pairs directly aligned with OmniFD's FF++ test list.

## Schema

- `image`: RGB human face image (`PIL.Image`)
- `mask`: Grayscale binary manipulation mask (`0` = authentic pixel, `255` = manipulated pixel)
- `label`: `0` (authentic) or `1` (tampered)
- `label_name`: `"authentic"` or `"tampered"`
- `domain_role`: split domain designation
- `source_dataset`: upstream source sub-dataset
- `manipulation_type`: manipulation family (`real_youtube`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `real_portrait`, `real_celebrity_frame`, `CelebDF_faceswap`)
- `image_id`: unique sample identifier
"""


def build_and_push(repo_id: str, save_local: Optional[str] = None, push: bool = True) -> None:
    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
    from huggingface_hub import HfApi

    print("Collecting FaceForensics++ C23 test splits (Model 1 & Model 2 In-Domain)...")
    m1_rows, m2_rows = _collect_ffpp_splits()
    print(f"  model1_in_domain: {len(m1_rows)} samples")
    print(f"  model2_in_domain: {len(m2_rows)} samples")

    print("Collecting Out-of-Domain human faces (CelebA-faces + Celeb-DF-v2)...")
    ood_rows = _collect_ood_split()
    print(f"  out_of_domain: {len(ood_rows)} samples")

    features = Features(
        {
            "image": HFImage(),
            "mask": HFImage(),
            "label": Value("int64"),
            "label_name": Value("string"),
            "domain_role": Value("string"),
            "source_dataset": Value("string"),
            "manipulation_type": Value("string"),
            "image_id": Value("string"),
        }
    )

    ds_dict = DatasetDict(
        {
            "model1_in_domain": Dataset.from_list(m1_rows, features=features),
            "model2_in_domain": Dataset.from_list(m2_rows, features=features),
            "out_of_domain": Dataset.from_list(ood_rows, features=features),
        }
    )

    if save_local:
        os.makedirs(save_local, exist_ok=True)
        ds_dict.save_to_disk(save_local)
        print(f"Saved local copy to {save_local}")

    if push:
        print(f"Pushing DatasetDict to Hugging Face Hub: {repo_id} ...")
        ds_dict.push_to_hub(repo_id, private=False)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=DATASET_CARD_MD.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card documenting Model 1/2 in-domain & out-of-domain face splits",
        )
        print(f"Published https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build & publish human-face domain dataset to Hugging Face Hub")
    p.add_argument("--repo-id", type=str, default="KhangTruong/face-forgery-domain-eval")
    p.add_argument("--save-local", type=str, default="outputs/face_domain_dataset")
    p.add_argument("--no-push", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_and_push(repo_id=args.repo_id, save_local=args.save_local, push=not args.no_push)
