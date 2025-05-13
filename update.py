import argparse
import json
from manager import load_lore_manager
from torch import optim
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct embeddings using a saved LoreManager via config file.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    args = parser.parse_args()

    # Load config from JSON
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract config
    model_path = config["model_path"]
    nodes = config["nodes"]
    relations = config["relations"]
    updates = config["updates"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lore_manager = load_lore_manager(device=device, path=model_path, optimizer_class=optim.Adam)

    lore_manager.update_kg(nodes, relations, updates)

    lore_manager.save_lore_manager(path=model_path)


