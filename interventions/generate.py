import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusion3Pipeline
from model_with_hooks import ModelWithHooks
from hooks import get_hook
import random
from typing import List
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, get_peft_model

def load_hooks(interventions_cache_dir: Path, intervention_name: str, device: str, block_indexes: List[int] = None):
    hooks = []
    for state_dict_path in interventions_cache_dir.glob("*.statedict"):
        print(f"Loading hook from {state_dict_path}")
        module_name = state_dict_path.stem  # Extract module name from file name

        # Extract block index from module_name
        parts = module_name.split('.')
        try:
            # Find the index of 'transformer_blocks' and get the following number
            idx = parts.index('transformer_blocks')
            block_idx = int(parts[idx + 1])

            # If specific block_indexes are provided, skip other blocks
            if block_indexes is not None and block_idx not in block_indexes:
                print(f"Skipping hook for block {block_idx} (not in block_indexes {block_indexes})")
                continue  # Skip this hook

        except (ValueError, IndexError) as e:
            print(f"Error parsing block index from module name '{module_name}': {e}")
            continue  # Skip this file if block index cannot be extracted

        # Set up the hook with the state dict
        hook = get_hook(
            intervention_name,
            module_name=module_name,
            device=device,
            state_path=state_dict_path  # Path directly passed to load state
        )
        print(f"Loaded hook for {module_name}")
        hooks.append(hook)
    return hooks

def load_prompts(file_path):
    """Helper function to load prompts from a file and format them."""
    with open(file_path, 'r') as file:
        return [f"A cat holding a sign '{line.strip()}'" for line in file if line.strip()]

def batchify(data, batch_size):
    """Helper function to split data into batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Toxicity Test")
    parser.add_argument('--toxic_file', type=str, required=True, help="Path to the file containing toxic prompts.")
    parser.add_argument('--non_toxic_file', type=str, required=True, help="Path to the file containing non-toxic prompts.")
    parser.add_argument('--toxic_output_dir', type=str, required=True, help="Directory to save images for toxic prompts.")
    parser.add_argument('--non_toxic_output_dir', type=str, required=True, help="Directory to save images for non-toxic prompts.")
    parser.add_argument('--interventions_dir', type=str, required=True, help="Path to interventions cache directory.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for processing prompts.")
    parser.add_argument('--timesteps', type=int, default=28, help="Number of inference steps.")
    parser.add_argument('--block_indexes', type=int, nargs='+', default=None, help="List of block indexes to focus on.")
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,

    )
    pipe.to("cuda")

    interventions_cache_dir = Path(args.interventions_dir)
    intervention_name = "det0"
    device = "cuda"

    hooks = load_hooks(
        interventions_cache_dir,
        intervention_name,
        device,
        block_indexes=args.block_indexes
    )
    print(f"Loaded {len(hooks)} hooks")
    pipe.model_with_hooks = ModelWithHooks(pipe.transformer, hooks=hooks)
    pipe.model_with_hooks.register_hooks()


    # peft_config = LoraConfig(
    #     r=16,  # or whatever you used in training
    #     lora_alpha=1,
    #     target_modules=["k_proj", "v_proj", "out_proj", "fc1", "fc2", "patch_embedding"],
    #     lora_dropout=0.1,
    #     bias="none",
    # )


    # # Load fine-tuned checkpoint
    # clipL_checkpoint_path = "/home/aditya/safe-clip/checkpoints_lambdas_2/job_187049_task_None_proc_0_pid_5878_time_1735956682/checkpoint_best_recall.pth"
    # clipG_checkpoint_path = "/home/aditya/safe-clip/checkpoints_lambdas_2/job_187048_task_None_proc_0_pid_882652_time_1735956692/checkpoint_best_recall.pth"

    # checkpoint_clipL= torch.load(clipL_checkpoint_path, map_location="cpu")
    # checkpoint_clipG = torch.load(clipG_checkpoint_path, map_location="cpu")

    # # Load pre-trained CLIP models
    # text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    # text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    # text_encoder_ft = get_peft_model(text_encoder, peft_config)
    # text_encoder_2_ft = get_peft_model(text_encoder_2, peft_config)
    # # Load fine-tuned weights into the models
    # text_encoder_ft.load_state_dict(checkpoint_clipL["text_encoder_ft"], strict=False)
    # text_encoder_2_ft.load_state_dict(checkpoint_clipG["text_encoder_ft"], strict=False)

    # text_encoder_ft.to("cuda", dtype=torch.float16)
    # text_encoder_2_ft.to("cuda", dtype=torch.float16)
    # pipe.text_encoder = text_encoder_ft
    # pipe.text_encoder_2 = text_encoder_2_ft
    # pipe.text_encoder_3 = None

    # Load toxic and non-toxic prompts
    toxic_prompts = load_prompts(args.toxic_file)
    non_toxic_prompts = load_prompts(args.non_toxic_file)
    
    # Output directory setup
    toxic_output_dir = Path(args.toxic_output_dir)
    toxic_output_dir.mkdir(parents=True, exist_ok=True)
    non_toxic_output_dir = Path(args.non_toxic_output_dir)
    non_toxic_output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
    # Generate images for toxic prompts
    for i, toxic_batch in enumerate(batchify(toxic_prompts, batch_size)):
        outputs = pipe(
            prompt=toxic_batch,
            num_inference_steps=args.timesteps,
            guidance_scale=7.0,
        )
        for idx, image in enumerate(outputs.images):
            image.save(toxic_output_dir / f"toxic_image_{i * batch_size + idx}.png")

    # Generate images for non-toxic prompts
    for i, non_toxic_batch in enumerate(batchify(non_toxic_prompts, batch_size)):
        outputs = pipe(
            prompt=non_toxic_batch,
            num_inference_steps=args.timesteps,
            guidance_scale=7.0,

        )
        for idx, image in enumerate(outputs.images):
            image.save(non_toxic_output_dir / f"non_toxic_image_{i * batch_size + idx}.png")

if __name__ == "__main__":
    main()
