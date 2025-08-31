from typing import Dict
from retrieval_metrics import compute_retrieval_metrics

def eval_split(
    name: str, split, model, processor, device,
    eval_bs_images: int, eval_bs_texts: int, is_siglip: bool
) -> Dict[str, float]:
    """
    Run retrieval metrics on a dataset split and pretty print a concise subset.
    """
    print(f"\n[Retrieval evaluation on {name}]")
    metrics = compute_retrieval_metrics(
        model=model,
        processor=processor,
        split_ds=split,
        device=device,
        eval_image_bs=eval_bs_images,
        eval_text_bs=eval_bs_texts,
        is_siglip=is_siglip,
    )
    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "i2t_MedR", "t2i_MedR", "avg_best_cosine", "n_images", "n_texts"]
    shown = {k: (round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]) for k in keys}
    print(shown)
    return metrics
