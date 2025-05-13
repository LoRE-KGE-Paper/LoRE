import torch


class LoreBatch:
    """
    A batch container for LoRE training.
    Holds node/relation indices, negative samples, and optional literal-augmented features.
    """

    def __init__(
        self,
        nodes,
        relations,
        subject_idxs,
        relation_idxs,
        object_idxs,
        subject_neg_idxs,
        object_neg_idxs,
        subject_mapping,
        object_mapping,
        lore_padded=None,
        lore_mask=None,
    ):
        self.nodes = nodes
        self.relations = relations
        self.subject_idxs = subject_idxs
        self.relation_idxs = relation_idxs
        self.object_idxs = object_idxs
        self.subject_neg_idxs = subject_neg_idxs
        self.object_neg_idxs = object_neg_idxs

        self.subject_mapping = subject_mapping
        self.object_mapping = object_mapping

        self.lore_padded = lore_padded
        self.lore_mask = lore_mask

    def to(self, device):
        """
        Moves all tensor-compatible attributes to the specified device (CPU or GPU).
        """
        self.nodes_tensor = torch.tensor(self.nodes, device=device)
        self.relations_tensor = torch.tensor(self.relations, device=device)
        self.subject_idx_tensor = torch.tensor(self.subject_idxs, device=device)
        self.relation_idx_tensor = torch.tensor(self.relation_idxs, device=device)
        self.object_idx_tensor = torch.tensor(self.object_idxs, device=device)
        self.subject_neg_idx_tensor = torch.tensor(self.subject_neg_idxs, device=device)
        self.object_neg_idx_tensor = torch.tensor(self.object_neg_idxs, device=device)

        if self.lore_padded is not None:
            self.lore_padded = [tensor.to(device) for tensor in self.lore_padded]
        if self.lore_mask is not None:
            self.lore_mask = [mask.to(device) for mask in self.lore_mask]
