import pickle
import random
import time
import zipfile
from collections import defaultdict
import copy
from datetime import datetime
from pathlib import Path
import os
from rdflib import Literal, Graph, XSD, URIRef
from batch import LoreBatch
from utils import BidirectionalMapping
from enum import Enum
import torch


def default_graph_node_factory(key):
    return GraphNode(key)


class DIR(Enum):
    OUT = -1
    IN = 1
    SELF = 0

    def inv(self):
        return DIR(-self.value)


class STAT(Enum):
    OK = 1
    DEPTH = 2
    EMPTY = 3
    KILL = 4
    KILLED = 5


class GraphNode:

    def __init__(self, idx=None, edges_out=None, edges_in=None, edges_self=None):
        self.edges_out = defaultdict(set, edges_out or {})
        self.edges_in = defaultdict(set, edges_in or {})
        self.edges_self = set() if edges_self is None else edges_self
        self.idx = idx

    def __sub__(self, other):
        if isinstance(other, GraphNode):
            # Define what happens when x - y is called
            new_graph_node = copy.deepcopy(self)

            for o_idx in new_graph_node.edges_out.keys() & other.edges_out.keys():
                new_graph_node.edges_out[o_idx].difference_update(other.edges_out[o_idx])
            for s_idx in new_graph_node.edges_in.keys() & other.edges_in.keys():
                new_graph_node.edges_in[s_idx].difference_update(other.edges_in[s_idx])

            new_graph_node.edges_self.difference_update(other.edges_self)

            return new_graph_node

        raise TypeError("Subtraction only supported between instances of GraphNode")

    def __add__(self, other):
        if isinstance(other, GraphNode):
            # Define what happens when x + y is called
            new_graph_node = copy.deepcopy(self)

            for o_idx in new_graph_node.edges_out.keys() & other.edges_out.keys():
                new_graph_node.edges_out[o_idx].update(other.edges_out[o_idx])
            for s_idx in new_graph_node.edges_in.keys() & other.edges_in.keys():
                new_graph_node.edges_in[s_idx].update(other.edges_in[s_idx])

            new_graph_node.edges_self.update(other.edges_self)

            return new_graph_node

        raise TypeError("Subtraction only supported between instances of GraphNode")

    def __len__(self):
        return self.len_out() + self.len_in() + self.len_self()

    def len_out(self):
        return sum(len(relations) for relations in self.edges_out.values())

    def len_in(self):
        return sum(len(relations) for relations in self.edges_in.values())

    def len_self(self):
        return len(self.edges_self)

    def add_edge(self, edge):

        s_idx, p_idx, o_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            raise ValueError("Neither subject nor object are equivalent to graph node identifier!")
        elif s_idx != self.idx:
            self.edges_in[s_idx].add(p_idx)
        elif o_idx != self.idx:
            self.edges_out[o_idx].add(p_idx)
        else:
            self.edges_self.add(p_idx)

    def check_exists(self, edge):

        s_idx, p_idx, o_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            return False
        elif s_idx != self.idx or o_idx != self.idx:
            if s_idx != self.idx:
                if s_idx not in self.edges_in:
                    return False
                if p_idx not in self.edges_in[s_idx]:
                    return False
            else:
                if o_idx not in self.edges_out:
                    return False
                if p_idx not in self.edges_out[o_idx]:
                    return False
        else:
            if p_idx not in self.edges_self:
                return False
        return True

    def remove_edge(self, edge):

        if not self.check_exists(edge):
            return 0

        s_idx, p_idx, o_idx = edge

        if s_idx != self.idx and o_idx != self.idx:
            raise ValueError("Neither subject nor object are equivalent to graph node identifier!")
        elif s_idx != self.idx or o_idx != self.idx:
            if s_idx != self.idx:
                try:
                    self.edges_in[s_idx].remove(p_idx)
                    if len(self.edges_in[s_idx]) == 0:
                        del self.edges_in[s_idx]
                except KeyError:
                    return 0
            else:
                try:
                    self.edges_out[o_idx].remove(p_idx)
                    if len(self.edges_out[o_idx]) == 0:
                        del self.edges_out[o_idx]
                except KeyError:
                    return 0
        else:
            try:
                self.edges_self.remove(p_idx)
            except KeyError:
                return 0

        return 1

    def iter(self, neighbour_idxs=None):
        for o_idx, p_idxs in self.edges_out.items():
            for p_idx in p_idxs:
                if neighbour_idxs:
                    if o_idx not in neighbour_idxs:
                        continue
                yield self.idx, p_idx, o_idx
        for p_idx in self.edges_self:
            yield self.idx, p_idx, self.idx


class GraphNodeDefaultDict(defaultdict):
    def __missing__(self, key):
        # Create a new value using the key
        self[key] = value = self.default_factory(key)
        return value


class LoreGraph:
    """
    Main class representing a graph with bidirectional mappings for nodes and relations.
    Supports edge manipulation, subgraph sampling, and (de)serialization.
    """

    def __init__(self, super_graph=None):

        self.node_mapping = BidirectionalMapping()
        self.relation_mapping = BidirectionalMapping()
        self.graph_nodes = GraphNodeDefaultDict(default_graph_node_factory)
        self.super_graph = super_graph
        self.num_edges = 0
        self.edges_rs = {}

    def populate_pickle(self, pickle_data):
        """Restore the graph from serialized pickle data."""

        print(f'Loading Graph from pickled data...')
        start = time.perf_counter()

        nodes_dict = pickle_data["nodes_dict"]
        relations_dict = pickle_data["relations_dict"]
        edges = pickle_data["edges"]

        for item in nodes_dict.keys():
            self.node_mapping.insert(item)
        for item in relations_dict.keys():
            self.relation_mapping.insert(item)

        num_edges = len(edges)

        # Restore edges
        for i, (s_idx, p_idx, o_idx) in enumerate(edges):
            if i % 1000 == 0 or i == len(edges) - 1:
                print(f"\rProgress: {i + 1}/{num_edges} edges inserted", end='', flush=True)
            self.add_edge((
                self.node_mapping.get_item(s_idx),
                self.relation_mapping.get_item(p_idx),
                self.node_mapping.get_item(o_idx)
            ))

        print('\nGraph loaded within', f"{(time.perf_counter() - start):.3f}", 'seconds.')

    def serialize(self, pickle_path=None):
        """Serialize the graph to a dictionary or optionally to a .pickle file."""

        nodes_dict = self.node_mapping.item2idx
        relations_dict = self.relation_mapping.item2idx

        edges = []
        for s_idx, node in self.graph_nodes.items():
            for o_idx, p_idxs in node.edges_out.items():
                for p_idx in p_idxs:
                    edges.append((s_idx, p_idx, o_idx))
            for p_idx in node.edges_self:
                edges.append((s_idx, p_idx, s_idx))

        out = {
            "nodes_dict": nodes_dict,
            "relations_dict": relations_dict,
            "edges": edges,
        }

        if pickle_path:
            with open(pickle_path, "wb") as f:
                pickle.dump(out, f)
            print(f"Graph saved to {pickle_path}")
        else:
            return out

    def populate_n3(self, n3_path):
        """Load and parse RDF triples from an .n3 file or zip archive and convert into graph."""

        print(f'Loading Graph from {n3_path}...')
        start = time.perf_counter()

        pickle_path = n3_path + ".pickle"

        if os.path.exists(pickle_path):
            print("Pickle file exists.")

            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            self.populate_pickle(data)
        else:
            print("Pickle file does not exist.")

            kg = Graph()
            path = Path(n3_path)

            if path.suffix == ".zip":
                n3_filename_in_zip = path.stem + ".n3"
                with zipfile.ZipFile(n3_path, 'r') as z:
                    with z.open(n3_filename_in_zip) as n3_file:
                        kg.parse(file=n3_file, format="n3")
            else:
                kg.parse(n3_path, format="n3")

            num_edges = len(kg)

            # Populate the raw_graph_dict using indexed nodes and relations
            for i, (s, p, o) in enumerate(kg):
                print(f"\rProgress: {i + 1}/{num_edges} edges inserted", end='', flush=True)

                if isinstance(o, Literal):
                    if o.datatype == XSD.boolean:
                        o = URIRef(str(p) + str(bool(o)))
                    else:
                        continue

                self.add_edge((str(s), str(p), str(o)))

            self.serialize(pickle_path=pickle_path)

            print('\nGraph loaded within', f"{(time.perf_counter() - start):.3f}", 'seconds.')

    def set_edges_rs(self, node_idxs=None):
        """Precompute directional edge sets for each node in the graph for downstream training."""

        self.edges_rs = {}

        if node_idxs is not None:
            nodes = [node for idx, node in self.graph_nodes.items() if idx in node_idxs]
        else:
            nodes = self.graph_nodes.values()

        for node in nodes:

            if node_idxs is not None:
                if node.idx not in node_idxs:
                    continue

            self_edges = (
                {node.idx: {(p_idx, DIR.SELF) for p_idx in node.edges_self}}
                if node.edges_self else {}
            )

            in_edges = {
                s_idx: {(p_idx, DIR.IN) for p_idx in p_idxs}
                for s_idx, p_idxs in node.edges_in.items()
                if p_idxs
            }

            out_edges = {
                o_idx: {(p_idx, DIR.OUT) for p_idx in p_idxs}
                for o_idx, p_idxs in node.edges_out.items()
                if p_idxs
            }

            combined = defaultdict(set)

            for d in [self_edges, in_edges, out_edges]:
                for idx, edges in d.items():
                    combined[idx].update(edges)

            self.edges_rs[node.idx] = dict(combined)

    def add_edge(self, edge):
        """Add a directed edge (s, p, o) to the graph with indexing."""

        s, p, o = edge

        self.node_mapping.insert(s)
        self.node_mapping.insert(o)
        self.relation_mapping.insert(p)

        s_idx = self.node_mapping.get_idx(s)
        o_idx = self.node_mapping.get_idx(o)
        p_idx = self.relation_mapping.get_idx(p)

        try:
            if p_idx in self.graph_nodes[s_idx].edges_out[o_idx]:
                return 0
        except KeyError:
            pass

        if s_idx == o_idx:
            self.graph_nodes[s_idx].add_edge(edge=(s_idx, p_idx, o_idx))
        else:
            self.graph_nodes[s_idx].add_edge(edge=(s_idx, p_idx, o_idx))
            self.graph_nodes[o_idx].add_edge(edge=(s_idx, p_idx, o_idx))

        self.num_edges += 1
        return 1

    def remove_edge(self, edge, item=False):
        """Remove a directed edge (s, p, o) from the graph."""

        s, p, o = edge

        if item:
            if s not in self.node_mapping.items():
                return 0
            if o not in self.node_mapping.items():
                return 0
            if p not in self.relation_mapping.items():
                return 0
            s = self.node_mapping.get_idx(s)
            p = self.relation_mapping.get_idx(p)
            o = self.node_mapping.get_idx(o)

        if s != o:
            out1 = self.graph_nodes[s].remove_edge((s, p, o))
            out2 = self.graph_nodes[o].remove_edge((s, p, o))
            if 1 == out1 == out2:
                out = 1
            elif 0 == out1 == out2:
                out = 0
            else:
                raise ValueError("Different removal results!")
        else:
            out = self.graph_nodes[s].remove_edge((s, p, o))

        self.num_edges -= out
        return out

    def __sub__(self, other):
        """Return a deep-copied graph with edges in `other` removed."""

        if isinstance(other, LoreGraph):

            new_lore_graph = copy.deepcopy(self)

            for graph_node in other.graph_nodes:
                for s_idx, p_idx, o_idx in graph_node.iter():
                    new_lore_graph.remove_edge((
                        other.node_mapping.get_item(s_idx),
                        other.relation_mapping.get_item(p_idx),
                        other.node_mapping.get_item(o_idx)
                    ), item=True)

            return new_lore_graph

        raise TypeError("Subtraction only supported between instances of LoreGraph")

    def __add__(self, other):
        """Return a deep-copied graph with edges from `other` added."""

        if isinstance(other, LoreGraph):

            new_lore_graph = copy.deepcopy(self)

            for graph_node in other.graph_nodes:
                for s_idx, p_idx, o_idx in graph_node.iter():
                    new_lore_graph.add_edge((
                        other.node_mapping.get_item(s_idx),
                        other.relation_mapping.get_item(p_idx),
                        other.node_mapping.get_item(o_idx)
                    ))

            return new_lore_graph

        raise TypeError("Subtraction only supported between instances of LoreGraph")

    def get_random_subgraph(self,
                            parent_start_idx,
                            node_weights,
                            max_edges=256,
                            max_nodes=64,
                            max_degree=16,
                            max_depth=5,
                            spawn_rate=0.2):

        """Returns a random subgraph which is the basis for a subgraph batch."""

        if parent_start_idx not in self.node_mapping.idx2item.keys():
            raise Exception(f'Node with idx {parent_start_idx} is unknown!')

        class DepthNode:
            def __init__(self, node, edges_rs, delta, depth=None, killed=None, inserted=None):

                start = datetime.now()
                self.depth = depth if depth is not None else max_depth + 1
                killed = killed if killed is not None else set()
                inserted = inserted if inserted is not None else set()
                keep = inserted - killed

                edges_rs = edges_rs[node.idx]
                self.edges = {idx: edges_rs[idx].copy() for idx in keep if idx in edges_rs}
                needed = max(0, max_degree - len(self.edges))

                if needed > 0:
                    candidates = [idx for idx in edges_rs.keys() if idx not in inserted | killed]
                    if candidates:
                        if len(candidates) > needed:
                            sample = random.sample(candidates, min(needed, len(candidates)))
                            for s in sample:
                                self.edges[int(s)] = edges_rs[int(s)].copy()
                        else:
                            for s in candidates:
                                self.edges[s] = edges_rs[s].copy()

                self.depth = depth if depth is not None else max_depth + 1
                self.edge_counter = 0
                self.idx = node.idx

                self._killed = False

                end = datetime.now()

                delta.append(end - start)

            def __len__(self):
                return sum([len(x) for x in self.edges.values()])

            def add(self, new_node):
                new_idx = new_node.idx
                if new_idx not in self.edges:
                    self.edges[new_idx] = {(rel, dir.inv()) for rel, dir in new_node.edges[self.idx]}
                    if new_node.depth + 1 < self.depth:
                        self.depth = new_node.depth + 1

            def status(self):
                if self._killed:
                    return STAT.KILLED
                if len(self) <= 0 or self.edge_counter >= max_degree:
                    return STAT.KILL
                if self.depth >= max_depth:
                    return STAT.DEPTH
                return STAT.OK

            def kill(self, idx):
                if idx == self.idx:
                    self._killed = True
                self.edges.pop(idx, None)

            def update(self, neighbour_idx, edge, depth, stepped=False):
                if self.depth > depth + 1:
                    self.depth = depth + 1
                if neighbour_idx in self.edges:
                    rel, dir = edge
                    if stepped:
                        self.edges[neighbour_idx].discard((rel, dir))
                    else:
                        self.edges[neighbour_idx].discard((rel, dir.inv()))
                    if not self.edges[neighbour_idx]:
                        self.edges.pop(neighbour_idx)

                if self.edge_counter == max_degree:
                    pass
                self.edge_counter += 1

            def keep_idxs(self, idxs):
                self.edges = {idx: edges for idx, edges in self.edges.items() if idx in idxs}

            def step(self):
                candidates = [idx for idx in self.edges]
                weights = [node_weights[key] for key in candidates]
                neighbour_idx = random.choices(candidates, weights=weights, k=1)[0]
                return neighbour_idx, random.choice(list(self.edges[neighbour_idx]))

        subgraph = LoreGraph(self)
        subgraph.node_mapping.insert(item=parent_start_idx)
        current_idx = parent_start_idx

        delta = []

        killed = set()
        inserted = {current_idx}
        depth_nodes = {
            current_idx: DepthNode(node=self.graph_nodes[current_idx], depth=0, edges_rs=self.edges_rs, delta=delta,
                                   killed=killed)}

        def killer():
            while kill_list := [node for node in depth_nodes.values() if node.status() == STAT.KILL]:
                kill_me = kill_list[0]
                for idx, node in ((idx, node) for idx, node in depth_nodes.items() if idx not in killed):
                    node.kill(kill_me.idx)
                killed.add(kill_me.idx)

        while subgraph.num_edges < max_edges:

            dNode = depth_nodes[current_idx]
            neighbour_idx, edge = dNode.step()
            rel_idx, dir = edge

            if dir == DIR.OUT:
                subgraph.add_edge((dNode.idx, rel_idx, neighbour_idx))
            elif dir == DIR.IN:
                subgraph.add_edge((neighbour_idx, rel_idx, dNode.idx))
            else:
                subgraph.add_edge((dNode.idx, rel_idx, dNode.idx))

            if neighbour_idx in depth_nodes:
                nNode = depth_nodes[neighbour_idx]
                nNode.update(dNode.idx, edge, dNode.depth)
                dNode.update(neighbour_idx, edge, dNode.depth, stepped=True)
                if dNode.status() == STAT.KILL or nNode.status() == STAT.KILL:
                    killer()

            else:

                dNode.update(neighbour_idx, edge, dNode.depth, stepped=True)
                inserted.add(neighbour_idx)
                nNode = DepthNode(node=self.graph_nodes[neighbour_idx],
                                  depth=dNode.depth + 1,
                                  edges_rs=self.edges_rs,
                                  delta=delta,
                                  killed=killed,
                                  inserted=inserted)

                depth_nodes[neighbour_idx] = nNode
                nNode.update(dNode.idx, edge, dNode.depth)
                for node_idx in nNode.edges:
                    if node_idx in depth_nodes and node_idx != neighbour_idx:
                        depth_nodes[node_idx].add(nNode)
                if len(depth_nodes) >= max_nodes:
                    for node in depth_nodes.values():
                        node.keep_idxs(depth_nodes.keys())
                    killer()
                else:
                    if dNode.status() == STAT.KILL or nNode.status() == STAT.KILL:
                        killer()

            if nNode.status() != STAT.OK or random.random() < spawn_rate:
                node_depths = defaultdict(set)
                for idx, node in depth_nodes.items():
                    if node.status() == STAT.OK:
                        node_depths[node.depth].add(idx)
                if len(node_depths) == 0:
                    break
                depth_candidates = list(node_depths.keys())
                depth_weights = [0.5 ** depth for depth in depth_candidates]
                selected_depth = random.choices(depth_candidates, weights=depth_weights, k=1)[0]
                current_idx = random.choice(list(node_depths[selected_depth]))
            else:
                current_idx = neighbour_idx

        return subgraph

    def training_batch(self,
                       device,
                       parent_start_idx,
                       node_weights,
                       max_edges=256,
                       max_nodes=64,
                       max_degree=16,
                       max_depth=5,
                       spawn_rate=0.2):
        """Transforms a subgraph into a trainable LoRE batch by adding false heads and tails for global training"""

        subgraph = self.get_random_subgraph(parent_start_idx=parent_start_idx,
                                            node_weights=node_weights,
                                            max_edges=max_edges,
                                            max_nodes=max_nodes,
                                            max_degree=max_degree,
                                            max_depth=max_depth,
                                            spawn_rate=spawn_rate)

        subject_idxs = []
        subject_neg_idxs = []
        relation_idxs = []
        object_idxs = []
        object_neg_idxs = []

        neg_nodes = [node.idx for node in subgraph.graph_nodes.values()]

        subject_mapping = {i: list() for i in range(len(neg_nodes))}
        object_mapping = {i: list() for i in range(len(neg_nodes))}

        edge_counter = 0

        for subj_node in subgraph.graph_nodes.values():

            if len(subj_node) == 0:
                continue

            subj_idx = subj_node.idx

            # Self loops
            for rel_idx in subj_node.edges_self:

                subj_neg_item = None
                obj_neg_item = None
                rel_item = subgraph.relation_mapping.get_item(rel_idx)

                random.shuffle(neg_nodes)
                edges_check = self.graph_nodes[subgraph.node_mapping.get_item(subj_idx)].edges_in
                for candidate_idx in neg_nodes:
                    if candidate_idx == subj_idx and len(neg_nodes) > 2:
                        continue
                    candidate_item = subgraph.node_mapping.get_item(candidate_idx)
                    if candidate_item not in edges_check:
                        subj_neg_item = candidate_item
                        break
                    if rel_item not in edges_check[candidate_item]:
                        subj_neg_item = candidate_item
                        break

                if subj_neg_item is None:
                    continue

                random.shuffle(neg_nodes)
                edges_check = self.graph_nodes[subgraph.node_mapping.get_item(subj_idx)].edges_out
                for candidate_idx in neg_nodes:
                    if candidate_idx == subj_idx and len(neg_nodes) > 2:
                        continue
                    candidate_item = subgraph.node_mapping.get_item(candidate_idx)
                    if candidate_item not in edges_check:
                        obj_neg_item = candidate_item
                        break
                    if rel_item not in edges_check[candidate_item]:
                        obj_neg_item = candidate_item
                        break

                if obj_neg_item is None:
                    obj_neg_item = subgraph.node_mapping.get_item(random.choices(neg_nodes))

                subj_neg_idx = subgraph.node_mapping.get_idx(subj_neg_item)
                obj_neg_idx = subgraph.node_mapping.get_idx(obj_neg_item)

                subject_mapping[subj_idx].append(edge_counter)
                object_mapping[subj_idx].append(edge_counter)

                edge_counter += 1

                subject_idxs.append(subj_idx)
                subject_neg_idxs.append(subj_neg_idx)
                relation_idxs.append(rel_idx)
                object_idxs.append(subj_idx)
                object_neg_idxs.append(obj_neg_idx)

            # Regular edges
            for obj_idx, rel_idxs in subj_node.edges_out.items():

                for rel_idx in rel_idxs:
                    subj_neg_item = None
                    obj_neg_item = None
                    rel_item = subgraph.relation_mapping.get_item(rel_idx)

                    random.shuffle(neg_nodes)
                    edges_check = self.graph_nodes[subgraph.node_mapping.get_item(obj_idx)].edges_in
                    for candidate_idx in neg_nodes:
                        if candidate_idx == subj_idx and len(neg_nodes) > 2:
                            continue
                        candidate_item = subgraph.node_mapping.get_item(candidate_idx)
                        if candidate_item not in edges_check:
                            subj_neg_item = candidate_item
                            break
                        if rel_item not in edges_check[candidate_item]:
                            subj_neg_item = candidate_item
                            break

                    if subj_neg_item is None:
                        subj_neg_item = subgraph.node_mapping.get_item(random.choices(neg_nodes))

                    random.shuffle(neg_nodes)
                    edges_check = self.graph_nodes[subgraph.node_mapping.get_item(subj_idx)].edges_out
                    for candidate_idx in neg_nodes:
                        if candidate_idx == subj_idx and len(neg_nodes) > 2:
                            continue
                        candidate_item = subgraph.node_mapping.get_item(candidate_idx)
                        if candidate_item not in edges_check:
                            obj_neg_item = candidate_item
                            break
                        if rel_item not in edges_check[candidate_item]:
                            obj_neg_item = candidate_item
                            break

                    if obj_neg_item is None:
                        obj_neg_item = subgraph.node_mapping.get_item(random.choices(neg_nodes))

                    subj_neg_idx = subgraph.node_mapping.get_idx(subj_neg_item)
                    obj_neg_idx = subgraph.node_mapping.get_idx(obj_neg_item)

                    subject_mapping[subj_idx].append(edge_counter)
                    object_mapping[obj_idx].append(edge_counter)

                    edge_counter += 1

                    subject_idxs.append(subj_idx)
                    subject_neg_idxs.append(subj_neg_idx)
                    relation_idxs.append(rel_idx)
                    object_idxs.append(obj_idx)
                    object_neg_idxs.append(obj_neg_idx)

        N = len(list(subgraph.node_mapping.items()))
        M = len(relation_idxs)
        all_combined = [None] * N
        all_masks = [None] * N

        for i in range(N):
            indices_obj = list(object_mapping.get(i, []))
            indices_subj = [int(M + j) for j in subject_mapping.get(i, [])]
            indices = indices_obj + indices_subj
            all_combined[i] = torch.tensor(indices, dtype=torch.long, device=device)
            all_masks[i] = torch.zeros(len(indices), dtype=torch.bool, device=device)

        return LoreBatch(
            nodes=list(subgraph.node_mapping.items()),
            relations=list(subgraph.relation_mapping.items()),
            subject_idxs=subject_idxs,
            relation_idxs=relation_idxs,
            object_idxs=object_idxs,
            subject_neg_idxs=subject_neg_idxs,
            object_neg_idxs=object_neg_idxs,
            subject_mapping=subject_mapping,
            object_mapping=object_mapping,
            lore_padded=all_combined,
            lore_mask=all_masks
        )
