import argparse
import json
from manager import load_lore_manager
from torch import optim
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain embeddings using a saved LoreManager via config file.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from JSON
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract paths
    model_path = config["model_path"]

    lore_manager = load_lore_manager(device=device, path=model_path, optimizer_class=optim.Adam)

    batch_specs = config.get("batch_specs", None)
    base_nodes = config.get("base_nodes", None)
    num_workers = config.get("num_workers", 0)
    epochs = config.get("epochs", 10)
    exponent = config.get("exponent", 2)

    print(f"Training config:")
    print(f"  Reusing model: {model_path}")
    print(f"  Batch specs: {batch_specs}")
    print(f"  Epochs: {epochs}")
    print(f"  Using device: {device}")

    lore_manager.lore_training(model_path=model_path,
                               batch_specs=batch_specs,
                               base_nodes=base_nodes,
                               num_workers=num_workers,
                               epochs=epochs,
                               exponent=exponent)
