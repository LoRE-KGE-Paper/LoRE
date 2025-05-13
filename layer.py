from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math


class GeoModule(nn.Module, ABC):

    def __init__(self, entity_embeddings):
        super().__init__()
        self.entity_embeddings = entity_embeddings

    def forward(self, heads, relations, inverse, head_base=None, relation_base=None, dropout=None):
        return self.predict(heads, relations, inverse, head_base, relation_base, dropout)

    @abstractmethod
    def predict(self, heads, relations, inverse, head_base, relation_base, dropout=None):
        pass

    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def tail(self, tails, relations, tail_base, relation_base):
        pass


class TransE(GeoModule):
    def __init__(self, entity_embeddings, num_relations, norm=2, embed_dim=256):

        super().__init__(entity_embeddings=entity_embeddings)

        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.norm = norm

    def predict(self, heads, relations, inverse=None, head_base=None, relation_base=None, dropout=None):

        if head_base is not None:
            head_embeddings = self.entity_embeddings(head_base)[heads]
        else:
            head_embeddings = self.entity_embeddings(heads)

        if relation_base is not None:
            relation_embeddings = self.relation_embeddings(relation_base)[relations]
        else:
            relation_embeddings = self.relation_embeddings(relations)

        if dropout is not None:
            head_embeddings = dropout(head_embeddings)
            relation_embeddings = dropout(relation_embeddings)

        if inverse is not None:
            relation_embeddings[inverse] *= -1

        out = head_embeddings + relation_embeddings
        if self.norm:
            out = F.normalize(out, p=2, dim=1)

        return out

    def normalize(self):
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1)

    def tail(self, tails, relations, tail_base, relation_base):

        if tail_base is not None:
            tail_embeddings = self.entity_embeddings(tail_base)[tails]
        else:
            tail_embeddings = self.entity_embeddings(tails)

        return tail_embeddings


class TransR(GeoModule):
    def __init__(self, entity_embeddings, num_relations, norm=2, embed_dim=256):

        super().__init__(entity_embeddings=entity_embeddings)

        self.num_relations = num_relations

        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.relation_projection = nn.Parameter(torch.empty(num_relations, embed_dim, embed_dim))
        bound = 1 / math.sqrt(embed_dim)
        init.uniform_(self.relation_projection, -bound, bound)

        self.norm = norm

    def predict(self, heads, relations, inverse=None, head_base=None, relation_base=None, dropout=None):

        if head_base is not None:
            head_embeddings = self.entity_embeddings(head_base)[heads]
        else:
            head_embeddings = self.entity_embeddings(heads)

        if relation_base is not None:
            relation_embeddings = self.relation_embeddings(relation_base)[relations]
            relation_projections = self.relation_projection[relation_base[relations]]
        else:
            relation_embeddings = self.relation_embeddings(relations)
            relation_projections = self.relation_projection[relations]

        if dropout is not None:
            head_embeddings = dropout(head_embeddings)
            relation_embeddings = dropout(relation_embeddings)

        head_embeddings = F.normalize(torch.bmm(head_embeddings.unsqueeze(1), relation_projections).squeeze(1), p=2,
                                      dim=1)

        if inverse is not None:
            relation_embeddings[inverse] *= -1

        out = head_embeddings + relation_embeddings
        if self.norm:
            out = F.normalize(out, p=2, dim=1)

        return out

    def normalize(self):

        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1)

    def tail(self, tails, relations, tail_base, relation_base):

        if tail_base is not None:
            tail_embeddings = self.entity_embeddings(tail_base)[tails]
        else:
            tail_embeddings = self.entity_embeddings(tails)

        if relation_base is not None:
            relation_projections = self.relation_projection[relation_base[relations]]
        else:
            relation_projections = self.relation_projection[relations]

        return F.normalize(torch.bmm(tail_embeddings.unsqueeze(1), relation_projections).squeeze(1), p=2, dim=1)


class TransD(GeoModule):
    def __init__(self, entity_embeddings, num_relations, norm=2, embed_dim=256):

        super().__init__(entity_embeddings=entity_embeddings)

        self.relation_embeddings = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        self.relation_embeddings_p = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.relation_embeddings_p.weight)

        self.entity_embeddings_p = nn.Embedding(*self.entity_embeddings.weight.shape)
        nn.init.xavier_uniform_(self.entity_embeddings_p.weight)

        self.norm = norm

    def predict(self, heads, relations, inverse=None, head_base=None, relation_base=None, dropout=None):

        if head_base is not None:
            head_embeddings = self.entity_embeddings(head_base)[heads]
            head_embeddings_p = self.entity_embeddings_p(head_base)[heads]
        else:
            head_embeddings = self.entity_embeddings(heads)
            head_embeddings_p = self.entity_embeddings_p(heads)

        if relation_base is not None:
            relation_embeddings = self.relation_embeddings(relation_base)[relations]
            relation_embeddings_p = self.relation_embeddings_p(relation_base)[relations]
        else:
            relation_embeddings = self.relation_embeddings(relations)
            relation_embeddings_p = self.relation_embeddings_p(relations)

        if dropout is not None:
            head_embeddings = dropout(head_embeddings)
            relation_embeddings = dropout(relation_embeddings)
            head_embeddings_p = dropout(head_embeddings_p)
            relation_embeddings_p = dropout(relation_embeddings)

        if inverse is not None:
            relation_embeddings[inverse] *= -1

        out = F.normalize(
            relation_embeddings_p * torch.sum(head_embeddings_p * head_embeddings, dim=1)[:, None] + head_embeddings,
            p=2, dim=1
        ) + relation_embeddings

        if self.norm:
            out = F.normalize(out, p=2, dim=1)

        return out

    def normalize(self):

        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1)
        self.relation_embeddings_p.weight.data = F.normalize(
            self.relation_embeddings_p.weight.data, p=2, dim=1)
        self.entity_embeddings_p.weight.data = F.normalize(
            self.entity_embeddings_p.weight.data, p=2, dim=1)

    def tail(self, tails, relations, tail_base, relation_base):

        if tail_base is not None:
            tail_embeddings = self.entity_embeddings(tail_base)[tails]
            tail_embeddings_p = self.entity_embeddings_p(tail_base)[tails]
        else:
            tail_embeddings = self.entity_embeddings(tails)
            tail_embeddings_p = self.entity_embeddings_p(tails)

        if relation_base is not None:
            relation_embeddings_p = self.relation_embeddings_p(relation_base)[relations]
        else:
            relation_embeddings_p = self.relation_embeddings_p(relations)

        out = relation_embeddings_p * torch.sum(tail_embeddings_p * tail_embeddings, dim=1)[:, None] + tail_embeddings

        if self.norm:
            out = F.normalize(out, p=2, dim=1)

        return out


class BatchLoreAttentionLayer(nn.Module):
    def __init__(self, embed_dim):

        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj]:
            init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                init.zeros_(proj.bias)

    def forward(self, embeddings, padding_mask=None):

        Q = self.q_proj(embeddings)
        K = self.k_proj(embeddings)

        d = embeddings.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)  # [B, L, L]

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, embeddings)

        if padding_mask is not None:
            valid_mask = ~padding_mask
            attended = attended * valid_mask.unsqueeze(-1)
            summed = attended.sum(dim=1)
            counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = attended.mean(dim=1)

        return F.normalize(pooled, dim=-1)
