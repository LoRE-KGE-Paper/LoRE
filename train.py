from manager import LoreManager
import torch
import argparse
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run initial LoreModel training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON config
    with open(args.config, 'r') as f:
        config = json.load(f)

    n3_path = config.get("n3_path")

    embed_dim = config.get("embed_dim", 256)
    base_model = config.get("base_model", "TransE")
    loss_type = config.get("loss", "LocosMRL")
    loss_margin = config.get("loss_margin", 0.5)

    lore_manager = LoreManager(n3_path=n3_path,
                               device=device,
                               embed_dim=embed_dim,
                               base_type=base_model,
                               loss_type=loss_type,
                               loss_margin=loss_margin)

    task_id = config.get("task_id")
    model_path = config.get("model_path", f"out/{task_id}/model.pickle")

    output_dir = f"out/{task_id}"
    os.makedirs(output_dir, exist_ok=True)

    lore_manager.optimizer.zero_grad()

    batch_specs = config.get("batch_specs", None)
    base_nodes = config.get("base_nodes", None)
    num_workers = config.get("num_workers", 0)
    epochs = config.get("epochs", 10)
    exponent = config.get("exponent", 2)

    print(f"Training config:")
    print(f"  Task ID: {task_id}")
    print(f"  Batch specs: {batch_specs}")
    print(f"  Epochs: {epochs}")
    print(f"  Using device: {device}")

    lore_manager.lore_training(model_path=model_path,
                               batch_specs=batch_specs,
                               base_nodes=base_nodes,
                               num_workers=num_workers,
                               epochs=epochs,
                               exponent=exponent)
