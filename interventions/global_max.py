import torch
from pathlib import Path
import argparse

def compute_global_max(input_dir):
    """
    Computes the global max across timesteps for each module in the given input directory
    and saves the result in the same subdirectory.
    """
    input_dir = Path(input_dir)

    for toxicity_type in ["toxic", "non_toxic"]:  # Iterate over toxic and non-toxic
        print(f"Processing {toxicity_type}...")
        toxicity_dir = input_dir / toxicity_type

        for batch_dir in toxicity_dir.iterdir():  # Iterate over batch directories
            if not batch_dir.is_dir():
                continue

            for prompt_dir in batch_dir.iterdir():  # Iterate over prompt directories
                if not prompt_dir.is_dir():
                    continue

                for module_dir in prompt_dir.iterdir():  # Iterate over modules
                    if not module_dir.is_dir():
                        continue

                    print(f"Processing {module_dir}...")

                    # Stack all timesteps for the module
                    activations = []
                    for timestep_file in sorted(module_dir.glob("timestep_*.pt")):
                        activations.append(torch.load(timestep_file))

                    if not activations:
                        continue
                    
                    # Compute global max across timesteps
                    all_activations = torch.stack(activations)  # Shape: [num_timesteps, num_neurons]
                    global_max = torch.max(all_activations, dim=0).values  # Shape: [num_neurons]

                    # Save the global max tensor in the same module directory
                    output_path = module_dir / "global_max.pt"
                    torch.save(global_max, output_path)
                    print(f"Saved global max to {output_path}")

    print("Global max computation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global max across timesteps for activations")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the input directory containing activations"
    )
    args = parser.parse_args()

    compute_global_max(args.input_dir)
