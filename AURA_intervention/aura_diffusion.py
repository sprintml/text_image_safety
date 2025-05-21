import argparse
import logging
from pathlib import Path
import torch
from hooks import HOOK_REGISTRY
import re

# Define the intervention functions to calculate alpha values for each type
def aura_fn(auroc: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculates alpha for AURA intervention based on AUROC scores."""
    alpha = torch.ones_like(auroc, dtype=torch.float16)
    mask = auroc > threshold
    alpha[mask] = 1 - 2 * (auroc[mask] - threshold)
    return alpha

def damp_fn(auroc: torch.Tensor, threshold: float = 0.5, damp_alpha: float = 0.5) -> torch.Tensor:
    """Calculates alpha for Damp intervention."""
    alpha = torch.ones_like(auroc, dtype=torch.float16)
    mask = auroc > threshold
    alpha[mask] = damp_alpha
    return alpha

def det0_fn(auroc: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculates alpha for Det0 intervention, setting alpha to 0 for selected neurons."""
    alpha = torch.ones_like(auroc, dtype=torch.float16)
    mask = auroc > threshold
    alpha[mask] = 0
    print(mask.long().sum())
    return alpha

alpha_fn_map = {
    "aura": aura_fn,
    "damp": damp_fn,
    "det0": det0_fn,
}

def load_auroc_scores(auroc_scores_dir: Path, score_type: str = "global_max") -> dict:
    """Loads AUROC scores for each module, specifically loading the requested score type (e.g., rolling_max)."""
    auroc_per_module = {}
    pattern = re.compile(rf"(.+)_({score_type})\.pt$")  # Pattern to match the module name and score type
    
    for file_path in auroc_scores_dir.glob("*.pt"):
        match = pattern.search(file_path.name)
        if match:
            # Extract the module name without the score type suffix
            module_name = match.group(1)
            print(file_path)
            auroc_per_module[module_name] = torch.load(file_path)
            print(f"Loaded {score_type} AUROC for module: {module_name} with shape {auroc_per_module[module_name].shape} and dtype {auroc_per_module[module_name].dtype}")
            
    return auroc_per_module


def main(args):
    
    auroc_per_module = load_auroc_scores(Path(args.auroc_scores_dir), score_type="global_max")

    for module_name, auroc in auroc_per_module.items():
        auroc_per_module[module_name] = auroc.to(torch.float16)
    # Prepare output directory
    intervention_dir = Path(args.interventions_cache_dir) / args.tag / args.model_name
    intervention_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving hooks in {intervention_dir}")

    # Iterate over each module, create and save hooks with calculated alpha
    for module_name, auroc in auroc_per_module.items():
        logging.info(f"Creating and saving hook for module: {module_name}")
        print(f"Creating and saving hook for module: {module_name}")
        # Generate alpha values using the specified intervention function
        alpha = alpha_fn_map[args.intervention](auroc, threshold=args.auroc_threshold)

        # Create the hook with the calculated alpha values
        hook = HOOK_REGISTRY[args.intervention](
            module_name=module_name,
            alpha=alpha.to(args.device),
        )
        
        # Save the hook's state dictionary
        torch.save(hook.state_dict(), intervention_dir / f"{module_name}.statedict")
        logging.info(f"Saved hook state for {module_name} at {intervention_dir / f'{module_name}.statedict'}")

    logging.info("All hooks saved successfully.")

def get_parser():
    parser = argparse.ArgumentParser(description="Save hooks for AURA-based intervention.")
    parser.add_argument("--auroc-scores-dir", type=str, required=True, help="Directory containing AUROC scores.")
    parser.add_argument("--interventions-cache-dir", type=str, required=True, help="Directory to save hook state files.")
    parser.add_argument("--tag", type=str, required=True, help="Tag for naming the output directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for directory naming.")
    parser.add_argument("--intervention", type=str, choices=["aura", "damp", "det0"], required=True, help="Type of intervention.")
    parser.add_argument("--auroc_threshold", type=float, default=0, help="AUROC threshold for applying intervention.")
    parser.add_argument("--damp_alpha", type=float, default=0.5, help="Alpha value for the damp intervention.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to save alpha values on.")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
