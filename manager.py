import pickle
from collections import deque
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from layer import TransE, TransR, TransD, BatchLoreAttentionLayer
from loss import LocosMRL, L2MRL
from batch import LoreBatch
from graph import LoreGraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from math import ceil
from torch import optim
from utils import LoreIterableDataset


class LoreManager(nn.Module):
    """
        Main manager class for orchestrating the LoRE graph instance and the embedding process.
    """

    def __init__(self, n3_path, device, base_type, loss_type, embed_dim, drop_rate=0.2,
                 graph_init=None, loss_margin=None, embedding_buffer=1000, embed_rows=None):
        super().__init__()
        self.device = device
        self.graph = LoreGraph()
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.embedding_buffer = embedding_buffer
        if n3_path is not None:
            self.graph.populate_n3(n3_path=n3_path)
            self.entity_embeddings = nn.Embedding(
                len(self.graph.node_mapping.items()) + embedding_buffer,
                embed_dim)
            nn.init.normal_(self.entity_embeddings.weight, mean=0.0, std=1.0)
        elif graph_init is not None:
            self.graph.populate_pickle(pickle_data=graph_init)
            self.entity_embeddings = nn.Embedding(embed_rows, embed_dim)
            nn.init.normal_(self.entity_embeddings.weight, mean=0.0, std=1.0)
        else:
            raise Exception("You must either provide a n3 path or a pickled lore graph")

        self.dropout = nn.Dropout(p=drop_rate).to(device)

        if base_type == "TransE":
            base_model = TransE
        elif base_type == "TransR":
            base_model = TransR
        elif base_type == "TransD":
            base_model = TransD
        else:
            raise Exception(f"{base_type} is not a valid LoRE base model!")

        self.base_type = base_type

        self.geo_module = base_model(entity_embeddings=self.entity_embeddings,
                                     num_relations=len(self.graph.relation_mapping.items()),
                                     norm=2, embed_dim=embed_dim).to(device)

        if loss_type == "LocosMRL":
            loss = LocosMRL
        elif loss_type == "L2MRL":
            loss = L2MRL
        else:
            raise Exception(f"{loss_type} is not a valid LoRE loss!")

        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.attention = BatchLoreAttentionLayer(embed_dim=embed_dim).to(device)
        self.embed_loss = loss(margin=loss_margin)
        self.to(self.device)
        self.normalize_embeddings()
        self.optimizer = optim.Adam(self.parameters(), lr=0.00005, weight_decay=1e-5)

    def update_kg(self, nodes, relations, updates):
        """
        Method used for on-the-fly KG updates.
        """

        for relation in relations:
            if relation not in self.graph.relation_mapping.items():
                raise Exception(f"Relation {relation} is unknown!")

        new_nodes = []

        for node in nodes:
            if node not in self.graph.node_mapping.items():
                for s, _, o in updates["remove"]:
                    s = nodes[s]
                    o = nodes[o]
                    if s == node or o == node:
                        raise Exception(f"Tried to remove edge with unknown node {node}!")
                new_nodes.append(node)

        removed_edges = [0, len(updates["remove"])]
        for s, p, o in updates["remove"]:
            s = nodes[s]
            p = relations[p]
            o = nodes[o]
            removed_edges[0] += self.graph.remove_edge((s, p, o), item=True)

        added_edges = [0, len(updates["add"])]
        for s, p, o in updates["add"]:
            s = nodes[s]
            p = relations[p]
            o = nodes[o]
            added_edges[0] += self.graph.add_edge((s, p, o))

        self.graph.set_edges_rs(node_idxs=[self.graph.node_mapping.get_idx(x) for x in nodes])

        len_new = len(new_nodes)

        print(f"Removed {removed_edges} edges. Added {added_edges}.")

        if len_new > 0:

            len_emb = self.entity_embeddings.weight.shape[0]
            len_graph = len(self.graph.node_mapping.items())

            overflow = len_graph - len_emb

            if overflow > 0:
                print("Update overlow: Extending entity embeddings..")
                self.extend_entity_embeddings(num_new_rows=overflow + self.embedding_buffer)
                print("Entity embeddings successfully extended!")

            new_embedding_idxs = [self.graph.node_mapping.get_idx(x) for x in new_nodes]
            initial_embeddings = self.get_reconstruction(reconstruct_items=new_nodes)

            with torch.no_grad():
                indices = torch.tensor(new_embedding_idxs, device=self.device)
                self.entity_embeddings.weight[indices] = initial_embeddings

            print(f"\nInitialized new embeddings: {len(new_embedding_idxs)}.")

    def extend_entity_embeddings(self, num_new_rows: int):
        """
        Add new entity embeddings, initialized as normalized vectors.
        Preserves optimizer state and gradient for existing parameters.
        """

        old_weight = self.entity_embeddings.weight.data
        old_grad = self.entity_embeddings.weight.grad.detach().clone() \
            if self.entity_embeddings.weight.grad is not None else None
        D = old_weight.shape[1]

        # Create new normalized vectors
        new_rows = F.normalize(torch.randn(num_new_rows, D, device=self.device), dim=1)
        new_weight = torch.cat([old_weight, new_rows], dim=0)

        # Update num_embeddings and embedding parameter
        self.entity_embeddings.num_embeddings += num_new_rows
        with torch.no_grad():
            self.entity_embeddings.weight = nn.Parameter(new_weight)

        # Restore gradients
        if old_grad is not None:
            new_grad = torch.zeros_like(new_weight)
            new_grad[:old_grad.shape[0]] = old_grad
            self.entity_embeddings.weight.grad = new_grad

        # Update geo_module with new embedding reference
        self.geo_module.entity_embeddings = self.entity_embeddings

        # Add updated parameter to optimizer
        self.optimizer.add_param_group({'params': [self.entity_embeddings.weight]})

    def get_reconstruction(self, reconstruct_items=None, forbidden_patterns=None, batch_size=16):
        """
        Derives the LoRE reconstructions for given nodes/items to be reconstructed.
        """

        with torch.no_grad():
            if forbidden_patterns is None:
                forbidden_patterns = []

            if reconstruct_items is None:
                idxs = list(self.graph.node_mapping.idxs())
            else:
                idxs = [self.graph.node_mapping.get_idx(x) for x in reconstruct_items]

            total_nodes = len(idxs)
            all_outputs = []
            num_batches = ceil(len(idxs) / batch_size)

            for b in range(num_batches):
                batch_indices = idxs[b * batch_size: (b + 1) * batch_size]
                processed = b * batch_size + len(batch_indices)

                print(
                    f"\rReconstructing Batch {b + 1}/{num_batches} "
                    f"({processed}/{total_nodes} nodes)...",
                    end='',
                    flush=True
                )
                batch_embeddings = []
                batch_masks = []

                for idx in batch_indices:
                    relations_in = []
                    relations_out = []
                    adj = []

                    for adj_idx, rel_idxs in self.graph.graph_nodes[idx].edges_in.items():
                        for rel_idx in rel_idxs:
                            relations_in.append(rel_idx)
                            adj.append(adj_idx)

                    for adj_idx, rel_idxs in self.graph.graph_nodes[idx].edges_out.items():
                        for rel_idx in rel_idxs:
                            if len(forbidden_patterns) > 0:
                                s_item = self.graph.node_mapping.get_item(idx)
                                p_item = self.graph.relation_mapping.get_item(rel_idx)
                                o_item = self.graph.node_mapping.get_item(adj_idx)
                                if matches_patterns((s_item, p_item, o_item), forbidden_patterns):
                                    continue
                            relations_out.append(rel_idx)
                            adj.append(adj_idx)

                    heads = torch.tensor(adj, device=self.device)
                    inverse = None
                    if len(relations_in) > 0 and len(relations_out) > 0:
                        relations = torch.cat([
                            torch.tensor(relations_in, device=self.device),
                            torch.tensor(relations_out, device=self.device)],
                            dim=0)
                        inverse = torch.cat([
                            torch.zeros(len(relations_in), dtype=torch.bool, device=self.device),
                            torch.ones(len(relations_out), dtype=torch.bool, device=self.device)
                        ])
                    elif len(relations_in) > 0:
                        relations = torch.tensor(relations_in, device=self.device)
                    elif len(relations_out) > 0:
                        relations = torch.tensor(relations_out, device=self.device)
                        inverse = torch.ones(len(relations_out), dtype=torch.bool, device=self.device)
                    else:
                        raise Exception("Node has no neighbors")

                    out = self.geo_module(heads, relations, inverse)

                    batch_embeddings.append(out)
                    batch_masks.append(torch.zeros(out.size(0), dtype=torch.bool, device=self.device))

                # Pad the batch and apply attention
                padded = pad_sequence(batch_embeddings, batch_first=True)
                mask = pad_sequence(batch_masks, batch_first=True, padding_value=True)
                attended = self.attention(padded, mask)
                all_outputs.append(attended)

            return torch.cat(all_outputs, dim=0)

    def lore_training(self, model_path, batch_specs=None, base_nodes=None, num_workers=0, epochs=10, exponent=2):
        """
        Function used in the (re-)training process for updating the KG embeddings. If required, base nodes can be
        defined so that training is limited to subgraphs centered around these nodes.
        """

        node_items = list(self.graph.node_mapping.items())

        if base_nodes is None:
            base_nodes = node_items

        base_idxs = [self.graph.node_mapping.get_idx(item) for item in base_nodes]

        recent_losses = deque(maxlen=100)
        recent_geo_losses = deque(maxlen=100)
        recent_lore_losses = deque(maxlen=100)

        recent_batch_sizes = deque(maxlen=100)

        full_times = deque(maxlen=100)
        batch_creation_times = deque(maxlen=100)
        forward_times = deque(maxlen=100)
        loss_times = deque(maxlen=100)
        backward_times = deque(maxlen=100)

        # Training step
        self.graph.set_edges_rs()
        self.train()

        for epoch in range(epochs):

            dataset = LoreIterableDataset(
                lore_graph=self.graph,
                node_idxs=base_idxs[:],
                node_weights={
                    idx: len(node) ** exponent
                    for idx, node in self.graph.graph_nodes.items()
                },
                batch_specs=batch_specs
            )

            dataloader = DataLoader(
                dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=True
            )

            i = 0

            for idx, batch in dataloader:
                full_times_start = datetime.now()

                batch.to(self.device)

                batch_creation_times.append((datetime.now() - full_times_start).total_seconds())

                forward_start = datetime.now()
                geo_fp, lore_fp = self(batch)
                forward_times.append((datetime.now() - forward_start).total_seconds())

                loss_start = datetime.now()
                geo_loss, lore_loss = self.get_loss(
                    geo_fp=geo_fp,
                    lore_fp=lore_fp,
                    batch=batch
                )
                loss = geo_loss + lore_loss
                loss_times.append((datetime.now() - loss_start).total_seconds())

                backward_start = datetime.now()
                loss.backward()
                self.optimizer.step()
                self.normalize_embeddings()
                backward_times.append((datetime.now() - backward_start).total_seconds())

                loss_item = loss.item()
                recent_losses.append(loss_item)
                recent_loss = sum(recent_losses) / len(recent_losses)

                geo_loss_item = geo_loss.item()
                recent_geo_losses.append(geo_loss_item)
                recent_geo_loss = sum(recent_geo_losses) / len(recent_geo_losses)

                lore_loss_item = lore_loss.item()
                recent_lore_losses.append(lore_loss_item)
                recent_lore_loss = sum(recent_lore_losses) / len(recent_lore_losses)

                batch_size = len(batch.object_idx_tensor)
                recent_batch_sizes.append(batch_size)
                recent_batch_size = sum(recent_batch_sizes) / len(recent_batch_sizes)

                full_times.append((datetime.now() - full_times_start).total_seconds())

                recent_full_times = sum(full_times) / len(full_times)
                recent_batch_creation_times = sum(batch_creation_times) / len(batch_creation_times)
                recent_forward_times = sum(forward_times) / len(forward_times)
                recent_loss_times = sum(loss_times) / len(loss_times)
                recent_backward_times = sum(backward_times) / len(backward_times)

                print(
                    f"\rEpoch {epoch + 1}/{epochs}: {i + 1}/{int(1.5 * len(base_nodes))}, "
                    f"Loss: {recent_loss:.5f} ({recent_geo_loss:.5f}, {recent_lore_loss:.5f}), "
                    f"Batch Size {recent_batch_size:.5f}, ",
                    f"Time: {recent_full_times:.3f} ({recent_batch_creation_times:.3f}, {recent_forward_times:.3f}, ",
                    f"{recent_loss_times:.3f}, {recent_backward_times:.3f})",
                    end='',
                    flush=True
                )

                i += 1

        self.save_lore_manager(path=model_path)

    def forward(self, batch: LoreBatch):

        heads = torch.cat([batch.subject_idx_tensor, batch.object_idx_tensor], dim=0)
        relations = torch.cat([batch.relation_idx_tensor, batch.relation_idx_tensor], dim=0)

        inverse = torch.cat([
            torch.zeros(batch.relation_idx_tensor.size(0), dtype=torch.bool, device=self.device),
            torch.ones(batch.relation_idx_tensor.size(0), dtype=torch.bool, device=self.device)
        ])

        head_base = batch.nodes_tensor
        relation_base = batch.relations_tensor

        dropout = self.dropout

        geo_fp = self.geo_module(heads=heads, relations=relations, inverse=inverse, head_base=head_base,
                                 relation_base=relation_base, dropout=dropout)

        if self.device == "cuda":
            torch.cuda.synchronize()

        padded_embeddings = pad_sequence([geo_fp[x] for x in batch.lore_padded], batch_first=True)
        padding_mask = pad_sequence(batch.lore_mask, batch_first=True, padding_value=True).to(self.device)

        lore_fp = F.normalize(self.attention(padded_embeddings, padding_mask), p=2, dim=1)

        return geo_fp, lore_fp

    def normalize_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)
            self.geo_module.normalize()

    def get_loss(self, geo_fp, lore_fp, batch: LoreBatch, mean=True):

        geo_loss = self.get_geo_loss(geo_fp=geo_fp, batch=batch, mean=mean)
        lore_loss = self.get_lore_loss(lore_fp=lore_fp, batch=batch, mean=mean)

        return geo_loss, lore_loss

    def get_geo_loss(self, geo_fp, batch: LoreBatch, mean=True):

        if geo_fp is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        tail_base = batch.nodes_tensor
        relation_base = batch.relations_tensor
        relations = torch.cat([batch.relation_idx_tensor, batch.relation_idx_tensor], dim=0)

        tails_pos = torch.cat([batch.object_idx_tensor, batch.subject_idx_tensor], dim=0)
        tails_neg = torch.cat([batch.object_neg_idx_tensor, batch.subject_neg_idx_tensor], dim=0)

        target_pos_embeddings = self.geo_module.tail(tails=tails_pos,
                                                     relations=relations,
                                                     tail_base=tail_base,
                                                     relation_base=relation_base)

        target_neg_embeddings = self.geo_module.tail(tails=tails_neg,
                                                     relations=relations,
                                                     tail_base=tail_base,
                                                     relation_base=relation_base)

        raw_loss = self.embed_loss.compute(
            out=geo_fp,
            target_pos_embeddings=target_pos_embeddings,
            target_neg_embeddings=target_neg_embeddings
        )

        return raw_loss.mean() if mean else raw_loss

    def get_lore_loss(self, lore_fp, batch: LoreBatch, temperature=0.1, mean=True):
        # Temperature: lower → greedier; higher → more exploratory

        if lore_fp is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        embedding_base = self.entity_embeddings(batch.nodes_tensor)
        distance_matrix = torch.cdist(embedding_base, embedding_base, p=2)
        distance_matrix.fill_diagonal_(float('inf'))
        logits = -distance_matrix / temperature
        closest_indices = torch.distributions.Categorical(logits=logits).sample()

        raw_loss = self.embed_loss.compute(
            out=lore_fp,
            target_pos_embeddings=embedding_base,
            target_neg_embeddings=embedding_base[closest_indices]
        )

        return raw_loss.mean() if mean else raw_loss

    def save_embedding(self, path, items, forbidden_patterns=None, reconstruct=False):

        reconstruct_idxs = [self.graph.node_mapping.get_idx(item) for item in items]

        if reconstruct:
            ent_embeddings_numpy = self.get_reconstruction(
                forbidden_patterns=forbidden_patterns, reconstruct_items=items
            ).detach().cpu().numpy()
            np.savez(path, embeddings=ent_embeddings_numpy, identifiers=np.array(items, dtype=object))
        else:
            ent_embeddings_numpy = self.entity_embeddings(
                torch.tensor(reconstruct_idxs, device=self.device)).detach().cpu().numpy()
            np.savez(path, embeddings=ent_embeddings_numpy, identifiers=np.array(items, dtype=object))

    def save_lore_manager(self, path: str):
        model_state = self.state_dict()
        graph_state = self.graph.serialize()

        data = {"model_state_dict": model_state, "graph_state": graph_state, "embed_dim": self.embed_dim,
                "drop_rate": self.drop_rate, "base_type": self.base_type, "loss_type": self.loss_type,
                "loss_margin": self.loss_margin, "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer_lr": self.optimizer.param_groups[0]["lr"]}

        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"\nLoreManager + LoreGraph{' + Optimizer' if self.optimizer else ''} saved to {path}")


def load_lore_manager(path: str, device: torch.device, optimizer_class=None):
    with open(path, "rb") as f:
        data = pickle.load(f)

    lore_manager = LoreManager(
        n3_path=None,
        device=device,
        embed_dim=data['embed_dim'],
        drop_rate=data['drop_rate'],
        graph_init=data['graph_state'],
        base_type=data['base_type'],
        loss_type=data['loss_type'],
        loss_margin=data['loss_margin'],
        embed_rows=data['model_state_dict']['entity_embeddings.weight'].shape[0],
    )

    lore_manager.load_state_dict(data["model_state_dict"])
    lore_manager.to(device)
    lore_manager.eval()

    if optimizer_class and "optimizer_state_dict" in data:
        lr = data.get("optimizer_lr", 1e-3)  # fallback in case not saved
        optimizer = optimizer_class(lore_manager.parameters(), lr=lr)
        optimizer.load_state_dict(data["optimizer_state_dict"])
        print(f"Optimizer restored with lr={lr}")
        lore_manager.optimizer = optimizer

    return lore_manager


def matches_patterns(entry, patterns):
    for pattern in patterns:
        p_count = 0
        e_count = 0
        for e, p in zip(entry, pattern):
            if p is not None:
                p_count += 1
                if e == p:
                    e_count += 1
            if p_count > 0 and p_count == e_count:
                return True
    return False
