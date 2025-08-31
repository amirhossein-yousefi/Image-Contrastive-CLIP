from typing import List
import torch

def ranks_from_scores(scores: torch.Tensor, gt_indices_per_row: List[List[int]]) -> List[int]:
    """
    scores: (R, C), higher is better.
    gt_indices_per_row: list of lists of ground-truth column indices for each row.
    Returns: list of best (lowest) 1-based ranks per row.
    """
    order = torch.argsort(scores, dim=1, descending=True)  # (R, C)
    best_ranks = []
    for r, gts in enumerate(gt_indices_per_row):
        ranks_r = []
        row = order[r]
        for g in gts:
            pos = (row == g).nonzero(as_tuple=False)
            ranks_r.append(int(pos[0, 0]) + 1)  # 1-based
        best_ranks.append(min(ranks_r))
    return best_ranks

def recall_at_k(best_ranks: List[int], k: int) -> float:
    return sum(1 for r in best_ranks if r <= k) / len(best_ranks)
