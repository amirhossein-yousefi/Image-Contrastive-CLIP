from typing import Dict
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from clip_utils import embed_images, embed_texts
from index_utils import build_eval_index
from rank_utils import ranks_from_scores, recall_at_k

def _device_is_cuda(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.startswith("cuda")
    return False

def compute_retrieval_metrics(
    model, processor, split_ds, device,
    eval_image_bs: int = 128, eval_text_bs: int = 256, is_siglip: bool = False,
    stream_if_products_over: float = 2e8,  # switch to streaming if N*M > 2e8
    chunk_size_text: int = 8192,
    chunk_size_image: int = 2048,
) -> Dict[str, float]:
    """
    Compute bi-directional retrieval metrics (image->text, text->image).
    Defaults preserved from the original function.
    """
    images, texts, img_to_txt, txt_to_img = build_eval_index(split_ds)

    # Embed
    img_emb = embed_images(model, processor, images, device, batch_size=eval_image_bs)  # (N, D)
    txt_emb = embed_texts(model, processor, texts,  device, batch_size=eval_text_bs, is_siglip=is_siglip)  # (M, D)

    N, D = img_emb.shape
    M, _ = txt_emb.shape

    # Use fp16 on GPU for speed/memory; ranking is unaffected by precision
    if _device_is_cuda(device):
        img_emb = img_emb.half()
        txt_emb = txt_emb.half()

    total_products = N * M
    if total_products <= stream_if_products_over:
        # -------- Small enough: compute full matrices --------
        sim_i2t = img_emb @ txt_emb.t()
        sim_t2i = txt_emb @ img_emb.t()

        i2t_best = ranks_from_scores(sim_i2t, img_to_txt)
        gt_img_idx_per_text = [[idx] for idx in txt_to_img]
        t2i_best = ranks_from_scores(sim_t2i, gt_img_idx_per_text)

    else:
        # -------- Streaming exact ranks (constant memory) --------
        i2t_best = []
        for i in tqdm(range(N), desc="i2t ranks (stream)", leave=False):
            v = img_emb[i : i + 1]  # (1,D)
            gts = img_to_txt[i]
            s_gt = (v @ txt_emb[gts].T).max().item()
            greater = 0
            for j in range(0, M, chunk_size_text):
                s = (v @ txt_emb[j : j + chunk_size_text].T).squeeze(0)  # (chunk,)
                greater += int((s > s_gt).sum().item())
            i2t_best.append(greater + 1)

        t2i_best = []
        for j in tqdm(range(M), desc="t2i ranks (stream)", leave=False):
            v = txt_emb[j : j + 1]  # (1,D)
            gt_img = txt_to_img[j]
            s_gt = float((v @ img_emb[gt_img : gt_img + 1].T).item())
            greater = 0
            for i in range(0, N, chunk_size_image):
                s = (v @ img_emb[i : i + chunk_size_image].T).squeeze(0)  # (chunk,)
                greater += int((s > s_gt).sum().item())
            t2i_best.append(greater + 1)

    # Metrics
    i2t_r1, i2t_r5, i2t_r10 = recall_at_k(i2t_best, 1), recall_at_k(i2t_best, 5), recall_at_k(i2t_best, 10)
    t2i_r1, t2i_r5, t2i_r10 = recall_at_k(t2i_best, 1), recall_at_k(t2i_best, 5), recall_at_k(t2i_best, 10)
    i2t_medr  = float(torch.tensor(i2t_best).median().item())
    t2i_medr  = float(torch.tensor(t2i_best).median().item())
    i2t_meanr = float(torch.tensor(i2t_best, dtype=torch.float32).mean().item())
    t2i_meanr = float(torch.tensor(t2i_best, dtype=torch.float32).mean().item())

    # Alignment quality: average of the best i2t GT similarity
    best_per_img = []
    for i in range(N):
        gts = img_to_txt[i]
        best_per_img.append(float((img_emb[i : i + 1] @ txt_emb[gts].T).max().item()))
    avg_best_cosine = sum(best_per_img) / len(best_per_img)

    return {
        "i2t_R@1": i2t_r1, "i2t_R@5": i2t_r5, "i2t_R@10": i2t_r10,
        "i2t_MedR": i2t_medr, "i2t_MeanR": i2t_meanr,
        "t2i_R@1": t2i_r1, "t2i_R@5": t2i_r5, "t2i_R@10": t2i_r10,
        "t2i_MedR": t2i_medr, "t2i_MeanR": t2i_meanr,
        "avg_best_cosine": avg_best_cosine,
        "n_images": N, "n_texts": M,
    }
