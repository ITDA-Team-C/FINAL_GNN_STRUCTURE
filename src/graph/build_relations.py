import os
import numpy as np
import pandas as pd
import torch
from src.utils import save_object, save_json
from src.graph.build_rur import build_rur
from src.graph.build_rtr import build_rtr
from src.graph.build_rsr import build_rsr
from src.graph.build_burst import build_burst
from src.graph.build_semsim import build_semsim
from src.graph.build_behavior import build_behavior

CONFIG = {
    "processed_dir": "data/processed",
    "nodes_file": "node_samples.csv",
    "features_file": "features.npy",
}


def load_data():
    nodes_path = os.path.join(CONFIG["processed_dir"], CONFIG["nodes_file"])
    features_path = os.path.join(CONFIG["processed_dir"], CONFIG["features_file"])

    df = pd.read_csv(nodes_path)
    features = np.load(features_path)

    print(f"[Load] 노드 수: {len(df)}")
    print(f"[Load] Feature 형태: {features.shape}")

    return df, features


def build_all_relations(df, features):
    print("\n[Graph] 모든 Relation 구성 중...\n")

    edge_index_dict = {}

    print("=" * 60)
    edge_index_dict["rur"] = build_rur(df)
    print()

    print("=" * 60)
    edge_index_dict["rtr"] = build_rtr(df)
    print()

    print("=" * 60)
    edge_index_dict["rsr"] = build_rsr(df)
    print()

    print("=" * 60)
    edge_index_dict["burst"] = build_burst(df)
    print()

    text_embeddings = features[:, :128]

    print("=" * 60)
    edge_index_dict["semsim"] = build_semsim(df, text_embeddings)
    print()

    print("=" * 60)
    edge_index_dict["behavior"] = build_behavior(df)
    print()

    return edge_index_dict


def convert_to_undirected(edge_index):
    if edge_index.shape[1] == 0:
        return edge_index

    forward = edge_index
    backward = torch.stack([edge_index[1], edge_index[0]])

    undirected = torch.cat([forward, backward], dim=1)
    undirected = torch.unique(undirected, dim=1)

    return undirected


def print_statistics(edge_index_dict):
    print("\n" + "=" * 60)
    print("[Summary] Relation 통계")
    print("=" * 60)

    total_edges = 0
    for relation_name, edge_index in edge_index_dict.items():
        num_edges = edge_index.shape[1]
        total_edges += num_edges
        print(f"{relation_name:15s}: {num_edges:8d} edges")

    print("-" * 60)
    print(f"{'Total':15s}: {total_edges:8d} edges")
    print("=" * 60)

    return total_edges


def save_relations(edge_index_dict, df):
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    print("\n[Save] Relation 저장 중...")

    for relation_name, edge_index in edge_index_dict.items():
        edge_path = os.path.join(CONFIG["processed_dir"], f"edge_index_{relation_name}.pt")
        torch.save(edge_index, edge_path)
        print(f"  {edge_path}")

    edge_dict_path = os.path.join(CONFIG["processed_dir"], "edge_index_dict.pt")
    torch.save(edge_index_dict, edge_dict_path)
    print(f"  {edge_dict_path}")

    graph_meta = {
        "num_nodes": len(df),
        "num_relations": len(edge_index_dict),
        "relation_names": list(edge_index_dict.keys()),
        "num_edges": {k: v.shape[1] for k, v in edge_index_dict.items()},
        "total_edges": sum(v.shape[1] for v in edge_index_dict.values()),
    }
    meta_path = os.path.join(CONFIG["processed_dir"], "graph_meta.json")
    save_json(graph_meta, meta_path)
    print(f"  {meta_path}")


if __name__ == "__main__":
    df, features = load_data()

    edge_index_dict = build_all_relations(df, features)

    for relation_name in edge_index_dict:
        edge_index_dict[relation_name] = convert_to_undirected(edge_index_dict[relation_name])

    total_edges = print_statistics(edge_index_dict)

    save_relations(edge_index_dict, df)

    print("\n[Done] Phase 3: Graph Construction 완료")
