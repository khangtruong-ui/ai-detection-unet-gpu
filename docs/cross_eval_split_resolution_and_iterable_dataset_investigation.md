# Investigation & Fix: Cross-Evaluation Split Resolution and IterableDataset Handling

## 1. Executive Summary

During execution of cross-dataset evaluation:
```bash
sid-cross-eval --cross-configs configs/cross-eval/MVSS-v1-protocol.yaml --checkpoint outputs/RUN/default/checkpoints/checkpoint_best.pt
```
Cross-evaluation aborted with two errors in `.logs.txt`:
1. `parquet-dataset-loader could not load 'KhangTruong/Tampered-Images-Combined-Dataset-Fake' (SplitNotFoundError: Split 'train' not found. Available splits: ['cross_test'])`
2. `TypeError: object of type 'IterableDataset' has no len()` in `sid_unet/dataset/loader.py` line 1343 (`return len(self.data)`).

This investigation answers the user's questions:
- **Did the YAML specify the train subset?** No. `configs/cross-eval/MVSS-v1-protocol.yaml` explicitly specified `eval_split: "cross_test"` and `test_split: "cross_test"`.
- **Was the cross-eval config not overridden well?** The configuration loading and overrides worked correctly.
- **Why was `Split 'train' not found` raised?** Because the candidate split fallback mechanism in `load_parquet_dataset` swallowed an `ImportError` on `cross_test` and iterated through all fallback split candidates (`cross-test`, `test`, `testing`, `validation`, `val`, `eval`, `train`), finally reporting the `SplitNotFoundError` of the last attempted candidate (`train`).

---

## 2. Root Cause Analysis

### 2.1 Why was split 'train' reported?

1. Remote Parquet streaming via `parquet-dataset-loader` uses `fsspec`'s `HTTPFileSystem` to perform HTTP Range requests against Hugging Face repositories.
2. In `fsspec`, `HTTPFileSystem` requires both `requests` and `aiohttp`.
3. `aiohttp` was missing from the environment and was not declared in `pyproject.toml` dependencies.
4. When `pdl.load_dataset(path="KhangTruong/Tampered-Images-Combined-Dataset-Fake", split="cross_test", streaming=True)` was called:
   ```python
   ImportError: HTTPFileSystem requires "requests" and "aiohttp" to be installed
   ```
5. In `sid_unet.dataset.loader.load_parquet_dataset()`, split candidates are resolved via:
   ```python
   get_split_candidates("cross_test") -> ["cross_test", "cross-test", "test", "testing", "validation", "val", "eval", "train"]
   ```
6. The candidate loop had an unconstrained exception handler:
   ```python
   for cand in candidates:
       try:
           ds = pdl.load_dataset(..., split=cand)
           return ds, cand
       except Exception as e:
           last_error = e
           continue
   ```
   - When candidate 0 (`cross_test`) failed with `ImportError`, the exception was swallowed.
   - The loop then tried `"cross-test"`, `"test"`, `"validation"`, and finally `"train"`.
   - On `"train"`, `pdl` raised: `SplitNotFoundError: Split 'train' not found. Available splits: ['cross_test']`.
   - When the loop exited, `last_error` held the `train` failure, masking the real `ImportError` and falsely indicating that `train` had been requested.

### 2.2 Why did `TypeError: object of type 'IterableDataset' has no len()` occur?

1. Because `parquet-dataset-loader` failed, `load_parquet_or_hf_dataset` fell back to `load_hf_dataset_robust(dataset_name, requested_split=requested_split, streaming=streaming)`.
2. Because `streaming=True`, Hugging Face `load_dataset` returned a streaming `IterableDataset`.
3. In `SIDMapDataset.__init__`, the map-style dataset initialization had:
   ```python
   # If fallback loaded an IterableDataset from HuggingFace (not indexable),
   # reload using HF non-streaming so it becomes an indexable Dataset:
   if not hasattr(ds, "__getitem__"):
       ds, resolved = load_hf_dataset_robust(dataset_name, requested_split=split, streaming=False)
   ```
4. In Hugging Face `datasets` (versions 2.x and later), `IterableDataset` implements `__getitem__` for slicing (`ds[:10]`). Therefore, `hasattr(ds, "__getitem__")` was `True`, causing the fallback check to evaluate to `False`.
5. As a result, `self.data` remained an `IterableDataset`.
6. When `DataLoader` or `__len__` queried `len(eval_dataset)`, Python raised:
   ```python
   TypeError: object of type 'IterableDataset' has no len()
   ```

---

## 3. Implemented Fixes

### 3.1 Dependency Declaration (`pyproject.toml`)
Added `aiohttp>=3.9.0` to `dependencies` in `pyproject.toml` so that HTTP range streaming dependencies for `parquet-dataset-loader` are automatically satisfied upon install.

### 3.2 Preserved Split Error Reporting (`sid_unet/dataset/loader.py`)
- In `load_parquet_dataset` and `load_hf_dataset_robust`:
  - Track `primary_error` on the first candidate (the actual `requested_split`).
  - If all candidates fail, re-raise `primary_error` so the error message always references the requested split (`cross_test`), preventing confusing errors about `'train'`.
  - Re-raise `(ImportError, ModuleNotFoundError)` immediately so missing environmental dependencies fail fast and are never masked by subsequent split candidate queries.

### 3.3 Fast-Fail Dependency Detection in `load_parquet_or_hf_dataset`
If `load_parquet_dataset` raises `(ImportError, ModuleNotFoundError)`, log and re-raise immediately, surfacing missing packages rather than silently falling back to slow multi-gigabyte Hugging Face downloads.

### 3.4 Robust `IterableDataset` Detection in `SIDMapDataset`
- Updated the guard in `SIDMapDataset.__init__`:
  ```python
  if isinstance(ds, (IterableDataset, HFIterableDataset)) or not hasattr(ds, "__len__"):
      ds, resolved = load_hf_dataset_robust(dataset_name, requested_split=split, streaming=False)
      self.resolved_split = resolved
      self.split = resolved
  ```
- Added safety fallbacks in `SIDMapDataset.__len__` to handle `num_rows` and `max_samples` if `len(self.data)` fails.

---

## 4. Verification & Testing

### 4.1 Unit & Integration Tests
- **`tests/test_parquet_loader_integration.py`**:
  - `test_load_parquet_dataset_fail_fast_on_import_error`: verified that missing dependencies raise `ImportError` on the first candidate without iterating to `train`.
  - `test_load_parquet_dataset_preserves_requested_split_primary_error`: verified that candidate split failures report `cross_test` rather than `train`.
  - `test_sid_map_dataset_hf_iterable_fallback_handling`: verified that `IterableDataset` from fallback triggers non-streaming reload and provides valid `len()`.
  - 13/13 tests **PASSED** in 14.8s.
- **`tests/test_cross_eval.py`**:
  - `test_cross_eval_cross_test_split_resolution_and_loader`: verified that `configs/cross-eval/MVSS-v1-protocol.yaml` loads and produces indexable batches on `cross_test` split without TypeError.
  - 14/14 tests **PASSED** in 56s.

### 4.2 End-to-End Live Execution
Executed `sid-cross-eval` with `configs/cross-eval/MVSS-v1-protocol.yaml` and `checkpoint_best.pt`:
- Peak VRAM: 299.6 MB
- Progress: 100% completed without any errors or unhandled exceptions.
- Master report and matrix successfully generated.
