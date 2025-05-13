from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class Loss(ABC):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    @abstractmethod
    def compute(self, out, target_pos_embeddings, target_neg_embeddings):
        """Subclasses implement this to compute the loss (before any reduction)."""
        pass


class LocosMRL(Loss):

    def __init__(self, margin=0.5):
        super().__init__(margin)

    def compute(self, out, target_pos_embeddings, target_neg_embeddings):
        pos_neg_center = (target_pos_embeddings + target_neg_embeddings) / 2
        cos_distance = 1 - F.cosine_similarity(out - pos_neg_center,
                                               target_pos_embeddings - pos_neg_center,
                                               dim=1)
        if self.margin:
            return torch.clamp(cos_distance - self.margin, min=0)
        else:
            return cos_distance


class L2MRL(Loss):

    def __init__(self, margin=0.5):
        super().__init__(margin)

    def compute(self, out, target_pos_embeddings, target_neg_embeddings):
        pos_distance = torch.norm(target_pos_embeddings - out, p=2, dim=1)
        neg_distance = torch.norm(out - target_neg_embeddings, p=2, dim=1)
        if self.margin:
            return torch.clamp(pos_distance + self.margin - neg_distance, min=0)
        else:
            return pos_distance
