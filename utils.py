import math
import random
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np


class BidirectionalMapping:
    def __init__(self, items=None):
        if items is None:
            items = []
        self.item2idx = {item: idx for idx, item in enumerate(items)}
        self.idx2item = {idx: item for idx, item in enumerate(items)}

    def get_idx(self, item):
        return self.item2idx.get(item)

    def contains(self, item):
        if item in self.item2idx.keys():
            return True
        return False

    def get_item(self, index):
        return self.idx2item.get(index)

    def items(self):
        return self.item2idx.keys()

    def idxs(self):
        return self.idx2item.keys()

    def insert(self, item):
        if item not in self.item2idx:
            next_idx = len(self.item2idx)
            self.item2idx[item] = next_idx
            self.idx2item[next_idx] = item

    def __len__(self):
        return len(self.item2idx)

    def __repr__(self):
        return f"BidirectionalMapping({len(self.item2idx)} items)"

    def __contains__(self, item):
        """Check if an object is a member of the class."""
        return item in self.item2idx


class LoreIterableDataset(IterableDataset):
    def __init__(self, lore_graph, node_idxs, node_weights, batch_specs):
        self.lore_graph = lore_graph
        self.node_idxs = create_augmented_list(node_idxs, [node_weights[x] for x in node_idxs])
        self.node_weights = node_weights
        self.batch_specs = batch_specs

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        local_indices = [idx for i, idx in enumerate(self.node_idxs) if i % num_workers == worker_id]

        for idx in local_indices:
            batch = self.lore_graph.training_batch(
                device="cpu",
                parent_start_idx=idx,
                node_weights=self.node_weights,
                **self.batch_specs
            )
            yield idx, batch


def weighted_shuffle(entries, weights):
    keys = [-np.log(random.random()) / w if w > 0 else float('inf') for w in weights]
    return [x for _, x in sorted(zip(keys, entries))]


def create_augmented_list(entries, weights):
    num_entries = len(entries)
    assert len(weights) == num_entries, "Lists 'a' and 'b' must be the same length"

    shuffled = weighted_shuffle(entries, weights)

    sample_size = math.ceil(num_entries / 2)
    sampled = random.choices(entries, weights=weights, k=sample_size)

    insert_positions = sorted(random.sample(range(num_entries + sample_size), k=sample_size))
    result = []
    shuffled_index = 0
    sample_index = 0

    for i in range(num_entries + sample_size):
        if sample_index < sample_size and i == insert_positions[sample_index]:
            result.append(sampled[sample_index])
            sample_index += 1
        else:
            result.append(shuffled[shuffled_index])
            shuffled_index += 1

    return result