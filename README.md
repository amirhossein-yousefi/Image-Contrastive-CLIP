# Image-Contrastive-CLIP
**Fine-tuning CLIP/SigLIP for image–text retrieval with Hugging Face Transformers**

> **TL;DR** — This repo provides a clean, reproducible reference for contrastive fine‑tuning and evaluation of **CLIP** (e.g., `openai/clip-vit-base-patch16`) and **SigLIP** (e.g., `google/siglip-base-patch16-224`) on public captioning datasets (Flickr8k/Flickr30k). It includes a custom `Trainer`, robust collators/tokenization for CLIP vs. SigLIP, retrieval metrics (R@K, MedR, cosine), and TensorBoard logging.

---

## Why this exists
Contrastive vision–language pretraining (à la **CLIP**) gives strong image–text representations out of the box. But **task‑adaptation** on retrieval datasets still buys measurable gains and serves as a good template for downstream finetunes. This project intentionally keeps the code modular and minimal—favoring **explicit data plumbing** and **evaluation hooks** over hidden magic—so you can adapt it to your own datasets or deploy it in production workflows.

---

## Repository structure

```
src/
  __init__.py
  callbacks.py           # Periodic retrieval evaluation during training (TrainerCallback)
  clip_utils.py          # Embedding helpers: image/text encoders + L2‑norm
  collators.py           # Batch builder; samples one caption per image (random) per step
  data_utils.py          # Flickr8k/30k dataset loaders via HF Datasets
  eval_utils.py          # Split‑level evaluation wrapper + pretty print
  evaluate_.py           # Scripted evaluation entrypoint (loads a checkpoint & dumps metrics)
  index_utils.py         # Build image↔text indices for retrieval
  main_training.py       # Training entrypoint (HF Trainer w/ contrastive loss)
  rank_utils.py          # Ranking helpers (image→text & text→image)
  retrieval_metrics.py   # R@1/5/10, MedR, average best cosine; n_images/n_texts
  trainer_custom.py      # Custom Trainer that computes CLIP/SigLIP contrastive loss
  runs/                  # Example experiment outputs (tensorboard logs + test_metric.json)
```
Key design choices surfaced in code:
- **Tokenizer/processors** are handled explicitly. For SigLIP, padding is set to `"max_length"` to match expected text shapes; for CLIP, dynamic padding is okay.
- **TF32** and **SDPA attention** are enabled where available for throughput on Ampere+ GPUs.
- **Random caption** per image per training step (a common recipe for retrieval finetuning) reduces overfitting and keeps batches well mixed.
- Evaluation metrics are computed **without side effects**; pure helpers return dictionaries that you can log or persist as JSON.

---

## Quickstart

### 1) Environment
Python ≥ 3.9 is recommended.
```bash
# (optional) conda
conda create -n ic-clip python=3.10 -y
conda activate ic-clip

# core deps
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick CUDA that matches your system
pip install -U transformers datasets accelerate timm pillow tqdm tensorboard

# (optional) retrieval / ANN acceleration
pip install faiss-cpu  # or faiss-gpu if you have CUDA & the right toolchain
```
<details>
<summary>Notes</summary>

- `accelerate` lets you scale to multi‑GPU seamlessly; the project also works with `torchrun`.
- `timm` is used for some model internals/weights in Transformers.
- If you are on CPU or Apple Silicon, install the appropriate PyTorch wheels.
</details>

### 2) Datasets
No manual download required—HF Datasets streams and caches for you.

- **Flickr8k**: `jxie/flickr8k` (5 captions per image)
- **Flickr30k**: `nlphuji/flickr30k` (Parquet branch is used to avoid legacy scripts)

Both loaders produce a `DatasetDict` with:
- `image` → PIL.Image
- `captions` → `List[str]`

---

## Training

> The training entrypoint is `src/main_training.py`. It uses a **custom `CLIPTrainer`** (in `trainer_custom.py`) built on top of `transformers.Trainer` to compute a CLIP‑style contrastive loss with proper temperature scaling and gather operations under DDP.

### Minimal single‑GPU run (CLIP on Flickr8k)
```bash
python -m src.main_training \
  --model_name openai/clip-vit-base-patch16 \
  --dataset flickr8k \
  --output_dir runs/clip-finetune-flickr8k \
  --epochs 5 \
  --lr 1e-5 \
  --train_bs 64 \
  --eval_bs 128 \
  --grad_accum 4 \
  --warmup_ratio 0.05 \
  --fp16
```

### SigLIP variant
```bash
python -m src.main_training \
  --model_name google/siglip-base-patch16-224 \
  --dataset flickr30k \
  --output_dir runs/siglip-finetune-flickr30k \
  --epochs 5 --lr 1e-5 --train_bs 64 --eval_bs 128 --grad_accum 4 --fp16
# Under the hood, the collator sets padding="max_length" for SigLIP text inputs.
```

### Multi‑GPU
With `torchrun`:
```bash
torchrun --nproc_per_node=4 -m src.main_training \
  --model_name openai/clip-vit-base-patch16 \
  --dataset flickr30k \
  --output_dir runs/clip-finetune-flickr30k \
  --epochs 5 --lr 1e-5 --train_bs 64 --eval_bs 128 --grad_accum 4 --fp16
```
Or with `accelerate`:
```bash
accelerate config  # first time only
accelerate launch -m src.main_training \
  --model_name openai/clip-vit-base-patch16 \
  --dataset flickr30k \
  --output_dir runs/clip-finetune-flickr30k \
  --epochs 5 --lr 1e-5 --train_bs 64 --eval_bs 128 --grad_accum 4 --fp16
```

### Memory safety & throughput
- Set `--image_resize 224` (or smaller like `196`) to reduce VRAM.
- `--grad_ckpt` enables gradient checkpointing on supported backbones.
- TF32 is enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) for **Ampere+** speedups.
- Attention uses SDPA (`attn_implementation="sdpa"`) when available.

---

## Evaluation

There are two ways to evaluate a checkpoint:

### A) Scripted evaluation (`src/evaluate_.py`)
Evaluate a **local or hub checkpoint** and write a JSON summary to your run directory.
```bash
python -m src.evaluate_ \
  --model_name /path/to/your/checkpoint_or_hub_id \
  --dataset flickr30k \
  --output_dir runs/clip-finetune-flickr30k \
  --eval_bs 128 --fp16
```
What it does:
- Loads the model/processor via `AutoModel`/`AutoProcessor`
- Builds an image–text index for the test split
- Computes retrieval metrics and writes `test_metric.json` under `runs/...`

> **Heads‑up**: The initial version of `evaluate_.py` contains a hard‑coded local Windows path for `AutoModel.from_pretrained`. Replace it with the `--model_name` argument (or a dedicated `--checkpoint` arg) to make the script portable. See the inline comments/TODO in the file.

### B) Programmatic evaluation
If you’re inside a notebook or another training script, call:
```python
from src.retrieval_metrics import compute_retrieval_metrics
from src.data_utils import load_image_text_dataset

ds = load_image_text_dataset("flickr8k")
metrics = compute_retrieval_metrics(model, processor, ds["test"], device,
                                    eval_image_bs=128, eval_text_bs=256,
                                    is_siglip=("siglip" in model.name_or_path.lower()))
print(metrics)  # dict with i2t_R@K, t2i_R@K, MedR, avg_best_cosine, n_images, n_texts
```

---

## What’s implemented

### Data pipeline
- **HF Datasets** loaders for Flickr8k/30k with **uniform schema** (`image`, `captions`)
- Casting images to `datasets.Image` (PIL)
- **Index building** (`index_utils.build_eval_index`) for efficient retrieval evaluation

### Tokenization & collators
- **Explicit** use of `processor.image_processor` and `tokenizer` subcomponents to avoid `CLIPProcessor.__call__` fast/slow path pitfalls
- **Random caption** selection per sample per iteration
- **SigLIP padding** set to `max_length` for transformer stability

### Model & training
- `AutoModel`/`AutoProcessor` support for **CLIP and SigLIP**
- **Custom `CLIPTrainer`**: computes contrastive loss, supports gradient checkpointing & mixed precision
- **Callback** (`RetrievalEvalCallback`) to periodically print R@K/MedR and cosine

### Evaluation & metrics
- **Bidirectional retrieval** metrics: image→text and text→image
- R@1/5/10, **Median Rank**, **avg_best_cosine**
- Batched embedding extraction with L2 normalization

### Logging
- **TensorBoard** logs saved under `runs/<exp-name>/logs`
```bash
tensorboard --logdir src/runs
```

---

## Extending to your own data
1. Package your dataset as a HF `DatasetDict` with columns:
   - `image` → PIL.Image
   - `captions` → `List[str]`
2. Add a loader alongside `load_image_text_dataset` or pass your dataset directly to the trainer/evaluator.
3. Ensure your text encoder is compatible (CLIP vs SigLIP tokenization & max length).

---

## Troubleshooting & FAQs

**OOM on 16GB GPUs**  
- Use `--image_resize 196` and `--train_bs 32 --grad_accum 8`  
- Enable `--grad_ckpt`  
- Prefer SigLIP on small batches; its sigmoid loss can be less batch‑size sensitive.

**Metrics look off**  
- Verify that you are evaluating the **same processor** as used in training.  
- Make sure captions are in a list (`List[str]`) and that you have mapped the dataset correctly (see `data_utils.py`).

**SigLIP only improves slightly**  
- Increase training steps/epochs; try a *small* learning rate schedule (e.g., warmup 5–10%).  
- Ensure text padding is `max_length` and that the tokenizer max length matches the checkpoint defaults.

**Multi‑GPU speed is low**  
- Use `torchrun` with NCCL backend.  
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  
- Confirm that SDPA attention is actually enabled (PyTorch ≥2.0).

---

## Reproducibility
- Seeding is centralized (`torch.manual_seed` and Python `random`) and applied early.
- TF32 is enabled to stabilize throughput on Ampere+; disable if you need bit‑wise determinism.
- Preprocessing parameters (resize/crop) can be **overridden** via CLI for explicit experiment tracking.

---

## Acknowledgements & References
- **CLIP**: Radford et al., *Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021.
- **SigLIP**: Google Research, *Sigmoid Loss for Language‑Image Pre‑training* (and SigLIP‑2 follow‑ups).  
- **Hugging Face**: `transformers`, `datasets` power most of this repo.

If you use this repo, consider citing CLIP/SigLIP and the datasets you finetune on.

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```

---

## License
As of the latest commit, no explicit license file is present in this repository. If you intend to distribute derivatives, please add a license (e.g., MIT/Apache‑2.0) or obtain permission from the author.

---

## Roadmap / TODO
- [ ] Make `evaluate_.py` accept a `--checkpoint` path instead of a hard‑coded local path
- [ ] Add `requirements.txt` and a proper `pyproject.toml`
- [ ] Provide example Weights & Biases logger hook (optional)
- [ ] Add tests for metrics and the collator edge cases

---

## Maintainer notes
- Experiments are saved under `src/runs/<exp-name>/`. Keep runs immutable—write new results to a new directory.
- Prefer **explicit** version pinning when moving to production (PyTorch, Transformers, Datasets).
- For ANN retrieval at scale, consider exporting embeddings and indexing with FAISS (HNSW/IVF‑PQ).

Happy training! 🧠📈
