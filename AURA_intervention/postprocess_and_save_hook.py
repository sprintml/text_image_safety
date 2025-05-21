import torch

class PostprocessAndSaveHook:
    def __init__(
        self,
        module_name,
        output_dir,
        save_fields=None,
        return_outputs=False,
    ):
        self.module_name = module_name
        self.output_dir = output_dir
        self.save_fields = save_fields or []
        self.return_outputs = return_outputs
        self.outputs = []

    def __call__(self, module, input, output):
        # Extract the desired outputs
        output_data = {}

        if isinstance(output, torch.Tensor):
            output_data["responses"] = output.detach().cpu()
        elif isinstance(output, tuple):
            for idx, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    output_data[f"output_{idx}"] = out.detach().cpu()

        # Store outputs if required
        if self.return_outputs:
            self.outputs.append(output_data)
            
        return output







# import torch
# from pathlib import Path
# from collections import defaultdict

# class PostprocessAndSaveHook(torch.nn.Module):
#     def __init__(
#         self,
#         module_name: str,
#         pooling_op_names: list,
#         output_path: Path,
#         save_fields: list,
#         return_outputs: bool = False,
#     ):
#         super().__init__()
#         self.module_name = module_name
#         self.pooling_ops = [lambda x: x]  # No pooling applied
#         self.output_path = output_path
#         self.save_fields = save_fields
#         self.return_outputs = return_outputs
#         self.outputs = defaultdict(list)

#     def save(self, module_name: str, output: dict, prompt_index: int, block_index: int) -> None:
#         # Extract timestep if it exists, otherwise raise an error
#         if "timestep" not in output:
#             raise ValueError("Timestep is required but was not provided in output.")
        
#         timestep = output["timestep"]
#         pooled_output = output["output"].detach().clone()

#         # Save the output and timestep with prompt and block indices for structured storage
#         datum = {
#             "responses": pooled_output.cpu(),
#             "timestep": timestep  # Save timestep with responses
#         }
#         # Create a structured file path based on module, prompt, block, and timestep
#         output_path = self.output_path / f"prompt_{prompt_index}" / f"block_{block_index}" / f"{module_name}_timestep_{timestep}.pt"
#         output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
#         torch.save(datum, output_path)

#     def __call__(self, module, input, output) -> None:
#         # Separate outputs by prompt_index to avoid combining them
#         batch_size = output.size(0)  # Get the batch size, assumed to match the number of prompts
#         print(batch_size)
#         for prompt_index in range(batch_size):
#             prompt_output = output[prompt_index]  # Get each prompt's output separately
#             # Store each prompt's output in `outputs` for callback access
#             self.outputs[self.module_name].append({"output": prompt_output})






