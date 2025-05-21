import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_global_max_and_labels(base_dir):
    """
    Loads global max activations and corresponding labels (toxic/non-toxic) by layer.

    Args:
        base_dir (Path): Base directory where global max activations are stored.

    Returns:
        global_max_activations (dict): Dictionary of global max activations by layer.
        labels (dict): Dictionary of labels corresponding to the activations.
    """
    global_max_activations = defaultdict(list)
    labels = defaultdict(list)

    for toxicity_type in ["toxic", "non_toxic"]:
        prompt_dir = base_dir / toxicity_type
        label = 1 if toxicity_type == "toxic" else 0

        for batch_folder in prompt_dir.glob("batch_*"):
            for prompt_folder in batch_folder.glob("prompt_*"):
                for module_folder in prompt_folder.glob("module_transformer_blocks_*"):
                    print(f"Processing module folder: {module_folder}")
                    for global_max_file in module_folder.glob("global_max.pt"):
                        # Transform the folder name correctly
                        layer_name = module_folder.name.replace("module_", "", 1).replace("blocks_", "blocks.").replace("_attn_", ".attn.")
                        if "_ff_net_0" in layer_name:
                            layer_name = layer_name.replace("_ff_net_0", ".ff.net.0")
                        if "_ff_net_1" in layer_name:
                            layer_name = layer_name.replace("_ff_net_1", ".ff.net.1")
                        if "_ff_net_2" in layer_name:
                            layer_name = layer_name.replace("_ff_net_2", ".ff.net.2")
                        if "_proj" in layer_name:
                            layer_name = layer_name.replace("_proj", ".proj")
                        print(f"Transformed layer name: {layer_name}")

                        activation = torch.load(global_max_file)

                        global_max_activations[layer_name].append(activation)
                        labels[layer_name].append(label)

    # Convert lists to tensors for each layer
    for layer_name in global_max_activations:
        global_max_activations[layer_name] = torch.stack(global_max_activations[layer_name])  # Shape: [num_samples, num_neurons]
        labels[layer_name] = torch.tensor(labels[layer_name])  # Shape: [num_samples]

    return global_max_activations, labels


def compute_auroc_from_global_max(global_max_activations, labels):
    """
    Computes AUROC scores for each neuron using global max activations.

    Args:
        global_max_activations (dict): Dictionary of global max activations by layer.
        labels (dict): Dictionary of labels corresponding to the activations.

    Returns:
        auroc_scores (dict): AUROC scores organized by layer.
    """
    auroc_scores = {}

    for layer_name, activations in global_max_activations.items():
        logging.info(f"Calculating AUROC for layer: {layer_name}")
        layer_labels = labels[layer_name]
        num_neurons = activations.shape[1]
        neuron_auroc_scores = []

        for neuron_idx in range(num_neurons):
            neuron_activations = activations[:, neuron_idx]
            try:
                auroc = roc_auc_score(layer_labels.numpy(), neuron_activations.numpy())
            except ValueError:
                # Default to 0.5 if AUROC cannot be calculated
                auroc = 0.5
            neuron_auroc_scores.append(auroc)
            logging.info(f"Layer: {layer_name}, Neuron: {neuron_idx}, AUROC: {auroc:.4f}")
        auroc_scores[layer_name] = neuron_auroc_scores
        #logging.info(f"Completed AUROC calculation for layer {layer_name}.")

    return auroc_scores


def main():
    parser = argparse.ArgumentParser(description="Compute AUROC scores using global max activations.")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory where global max activations are stored.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Step 1: Load global max activations and labels
    logging.info("Loading global max activations and labels...")
    global_max_activations, labels = load_global_max_and_labels(base_dir)

    # Step 2: Compute AUROC scores for each layer
    logging.info("Starting AUROC calculations...")
    auroc_scores = compute_auroc_from_global_max(global_max_activations, labels)

    # Step 3: Save AUROC scores
    output_dir = base_dir / "auroc_stats"
    output_dir.mkdir(exist_ok=True, parents=True)
    for layer_name, scores in auroc_scores.items():
        output_path = output_dir / f"{layer_name}_global_max.pt"
        torch.save(torch.tensor(scores), output_path)
        logging.info(f"AUROC scores for {layer_name} saved to {output_path}")


if __name__ == "__main__":
    main()
