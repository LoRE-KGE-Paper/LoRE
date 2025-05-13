import argparse
import json
from manager import load_lore_manager
import numpy as np
from torch import optim
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reconstruct embeddings using a saved LoreManager via config file.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    args = parser.parse_args()

    # Load config from JSON
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract paths
    model_path = config["model_path"]
    reconstruct_items = config["nodes"]
    out_path = config.get("out_path", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lore_manager = load_lore_manager(device=device, path=model_path, optimizer_class=optim.Adam)

    embedding_reconstructed = lore_manager.get_reconstruction(
        reconstruct_items=reconstruct_items).detach().cpu().numpy()

    if out_path:
        np.savez(out_path, embeddings=embedding_reconstructed,
                 identifiers=np.array(reconstruct_items, dtype=object))
