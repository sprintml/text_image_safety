import torch
from torch.utils.hooks import RemovableHandle

class ModelWithHooks:
    def __init__(self, module: torch.nn.Module, hooks=None) -> None:
        self.hooks = {h.module_name: h for h in hooks} if hooks is not None else {}
        self.module = module
        self._forward_hook_handles = []

    def register_hooks(self):
        # Register hooks by iterating over named_modules for flexible matching
        for module_name, module in self.module.named_modules():
            print(f"Checking for hooks for {module_name}")
            if module_name in self.hooks:
                print(f"Registering hook for {module_name}")
                hook_fn = self.hooks[module_name]
                self._forward_hook_handles.append(module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        for h in self._forward_hook_handles:
            h.remove()
        self._forward_hook_handles = []
        self.hooks = {}

    def get_hook_outputs(self):
        outputs = {name: hook.outputs for name, hook in self.hooks.items()}
        return outputs

    def clear_hook_outputs(self):
        for hook in self.hooks.values():
            hook.outputs.clear()