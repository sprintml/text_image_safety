import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
from model_with_hooks import ModelWithHooks
from postprocess_and_save_hook import PostprocessAndSaveHook
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
import argparse
import random
from datasets import load_dataset

def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output.values

class TorchPoolingOP(torch.nn.Module):
    def __init__(self, op_name: str, dim: int):
        super().__init__()
        self.name = op_name
        self.dim = dim
        if op_name == "max":
            self.op = lambda x: nanmax(x, dim=dim, keepdim=False)
        else:
            raise ValueError(f"Unsupported pooling operation: {op_name}")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.op(tensor)

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Define the step callback
def step_callback(pipeline, step, timestep, callback_kwargs):
    global batch_counter
    print(f"Step: {step}, Timestep: {timestep}, Batch: {batch_counter}")

    hook_outputs = pipeline.model_with_hooks.get_hook_outputs()

    for module_name, output_list in hook_outputs.items():
        if not output_list:
            continue  

        for output_idx, output_data in enumerate(output_list):
            responses = output_data.get("responses")
            if responses is None:
                continue
            
            # Select only conditioned activations
            batch_size = responses.shape[0] // 2  # Assuming batch size is doubled
            conditioned_responses = responses[batch_size:]
            pooled_responses = pooling_op(conditioned_responses)
            for prompt_index in range(batch_size):
                sanitized_module_name = module_name.replace(".", "_")
            
                output_path = (
                    output_dir / ("toxic" if is_toxic else "non_toxic") /
                    f"batch_{batch_counter}" / f"prompt_{prompt_index}" /
                    f"module_{sanitized_module_name}" /
                    f"timestep_{timestep}_output_{output_idx}.pt"
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pooled_responses[prompt_index].cpu(), output_path)

    pipeline.model_with_hooks.clear_hook_outputs()
    return callback_kwargs



def main(timesteps):

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load the pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    global output_dir, pooling_op, batch_counter, is_toxic
    output_dir = Path(f"sd3_cross_attention_layer10")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hook setup for monitoring cross-attention layers
    num_transformer_blocks = len(pipe.transformer.transformer_blocks)
    hook_modules = [
        f"transformer_blocks.10.attn.to_q"
    ] + [
        f"transformer_blocks.10.attn.to_k"
    ] + [
        f"transformer_blocks.10.attn.to_v"
    ]

    print(hook_modules)
    pipe.model_with_hooks = ModelWithHooks(
        pipe.transformer,
        hooks=[
            PostprocessAndSaveHook(
                module_name=name,
                output_dir=output_dir,
                save_fields=["responses"],
                return_outputs=True,
            )
            for name in hook_modules
        ],
    )
    pipe.model_with_hooks.register_hooks()
    pooling_op = TorchPoolingOP(op_name="max", dim=1)
    # Load ViSU dataset
    # visu_dataset = load_dataset("aimagelab/ViSU-Text")
    # nsfw_prompts = visu_dataset['train']['nsfw']
    # safe_prompts = visu_dataset['train']['safe']

    # # Randomly sample 1,000 rows
    # sampled_indices = random.sample(range(len(nsfw_prompts)), 1000)
    # toxic_prompts = [nsfw_prompts[i] for i in sampled_indices]
    # non_toxic_prompts = [safe_prompts[i] for i in sampled_indices]

    # print(f"Sampled toxic prompts: {toxic_prompts[:5]}")
    # print(f"Sampled non-toxic prompts: {non_toxic_prompts[:5]}")


    toxic_file_path = 'offensive_words.txt'
    non_toxic_file_path = 'all_unigram_1000_10000_100.txt'

    toxic_prompts = []
    non_toxic_prompts = []

    with open(toxic_file_path, 'r') as toxic_file:
        toxic_prompts = [f"A dog holding a sign '{line.strip()}'" for line in toxic_file if line.strip()]

    with open(non_toxic_file_path, 'r') as non_toxic_file:
        non_toxic_prompts = [f"A dog holding a sign '{line.strip()}'" for line in non_toxic_file if line.strip()]

    batch_size = 8

    # Run toxic prompts with classifier-free guidance
    is_toxic = True
    batch_counter = 0
    for toxic_batch in batchify(toxic_prompts, batch_size):
        pipe(
            prompt=toxic_batch,
            num_inference_steps=timesteps,
            callback_on_step_end=step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=7.0
        )
        batch_counter += 1

    # Run non-toxic prompts with classifier-free guidance
    is_toxic = False
    batch_counter = 0
    for non_toxic_batch in batchify(non_toxic_prompts, batch_size):
        pipe(
            prompt=non_toxic_batch,
            num_inference_steps=timesteps,
            callback_on_step_end=step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=7.0
        )
        batch_counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50, help="Number of time steps for diffusion")
    args = parser.parse_args()
    main(args.timesteps)







## MLP Layers

# [
#             f"transformer_blocks.{i}.ff.net.2" for i in range(num_transformer_blocks)
#         ] +[
#             f"transformer_blocks.{i}.ff.net.0" for i in range(num_transformer_blocks)
#         ] +[
#             f"transformer_blocks.{i}.ff.net.0.proj" for i in range(num_transformer_blocks)
#         ] +[
#             f"transformer_blocks.{i}.ff.net.1" for i in range(num_transformer_blocks)
#         ]