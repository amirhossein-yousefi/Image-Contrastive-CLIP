from transformers import Trainer

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # Ask model to compute its own loss (works for CLIP and SigLIP)
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss
