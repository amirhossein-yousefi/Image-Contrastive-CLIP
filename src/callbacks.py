from transformers import TrainerCallback
from retrieval_metrics import compute_retrieval_metrics

class RetrievalEvalCallback(TrainerCallback):
    """
    Periodically run retrieval evaluation on a provided split.
    Defaults preserved from your original callback wiring.
    """
    def __init__(self, processor, split_ds, eval_bs=128, is_siglip=False, every_n_epochs=1):
        self.processor = processor
        self.split_ds = split_ds
        self.eval_bs = eval_bs
        self.is_siglip = is_siglip
        self.every = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch or 0) % self.every != 0:
            return
        model = kwargs["model"]
        device = model.device
        metrics = compute_retrieval_metrics(
            model, self.processor, self.split_ds, device,
            eval_image_bs=self.eval_bs, eval_text_bs=max(128, self.eval_bs),
            is_siglip=self.is_siglip
        )
        print(f"\n[Validation retrieval @ epoch {int(state.epoch)}]")
        keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10",
                "i2t_MedR", "t2i_MedR", "avg_best_cosine"]
        print({k: (round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]) for k in keys})
