import torch
from .manager_modules import LinearLayerMemoryManager, ConvLayerMemoryManager
import random

LINEAR_MODULES = [
    "Linear",
    "LoRACompatibleLinear",
    "QLinear",
]
CONV_MODULES = [
    "Conv2d",
    "LoRACompatibleConv",
    "QConv2d",
]

UNMANAGED_MODULES = [
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "Embedding",
    "EmbeddingBag",
    "RNNBase",
    "LSTM",
    "GRU",
    "RNN",
    "Conv3d"
]

UNMANAGED_MODULES_INCLUDES = ["RotaryEmbedding", "Norm", "RotaryPosEmbed"]


class MemoryManager:
    def __init__(
        self,
        module: torch.nn.Module,
        process_device: torch.device = torch.device("cpu"),
    ):
        self.module: torch.nn.Module = module
        self.process_device: torch.device = process_device
        self.unmanaged_modules: list[torch.nn.Module] = []

    def memory_managed_to(self, *args, **kwargs):
        # first move all the unmanaged modules
        for module in self.unmanaged_modules:
            if isinstance(module, torch.nn.Parameter):
                # Parameter cannot move this way
                module.data = module.data.to(*args, **kwargs)
            else:
                module.to(*args, **kwargs)
        # check for a dtype argument
        dtype = None
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        elif len(args) > 0:
            for i, arg in enumerate(args):
                if isinstance(arg, torch.dtype):
                    dtype = arg
                    break
        if dtype is not None:
            return self.module._mm_to(dtype=dtype)
        return self.module

    @classmethod
    def attach(
        cls, 
        module: torch.nn.Module, 
        device: torch.device, 
        offload_percent: float = 1.0,
        ignore_modules: list[torch.nn.Module] = []
    ):
        if hasattr(module, "_memory_manager"):
            # already attached
            return

        module._memory_manager = cls(module, device)

        # override the to method to handle memory management
        module._mm_to = module.to
        module.to = module._memory_manager.memory_managed_to

        # add ignore modules to unmanaged list
        for im in ignore_modules:
            module._memory_manager.unmanaged_modules.append(im)
            
        # count ignore modules as processed
        modules_processed = [x for x in ignore_modules]

        # Scan the model and calculate the total number of target layers
        total_target_layers = 0
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if child_module in modules_processed:
                    continue
                if child_module.__class__.__name__ in LINEAR_MODULES or child_module.__class__.__name__ in CONV_MODULES:
                    total_target_layers += 1

        # Calculate how many layers are "privileged" to be kept on the GPU
        # e.g., for 100 layers and offload_percent=0.85, keep_on_gpu_count = 15
        keep_on_gpu_count = int(total_target_layers * (1.0 - offload_percent))
        current_layer_idx = 0

        # Sequential allocation (prioritize keeping the initial layers)
        # attach to all modules
        for name, sub_module in module.named_modules():
            for child_name, child_module in sub_module.named_modules():
                if child_module in modules_processed:
                    continue
                
                is_linear = child_module.__class__.__name__ in LINEAR_MODULES
                is_conv = child_module.__class__.__name__ in CONV_MODULES

                if is_linear or is_conv:
                    # Determine whether to stay on the GPU
                    skip = False 
                    if current_layer_idx < keep_on_gpu_count:
                        skip = True  # Do not apply offloading (keep on GPU) if within the quota
                    
                    current_layer_idx += 1

                    if skip:
                        module._memory_manager.unmanaged_modules.append(child_module)
                    else:
                        if is_linear:
                            LinearLayerMemoryManager.attach(child_module, module._memory_manager)
                        elif is_conv:
                            ConvLayerMemoryManager.attach(child_module, module._memory_manager)
                        
                        # attach to ARA (Accuracy Recovery Adapter) as well
                        if hasattr(child_module, "ara_lora_ref"):
                            ara = child_module.ara_lora_ref()
                            if ara not in modules_processed:
                                MemoryManager.attach(
                                    ara,
                                    device,
                                ) 
                            modules_processed.append(ara)
                    modules_processed.append(child_module)
                elif child_module.__class__.__name__ in UNMANAGED_MODULES or any(
                    inc in child_module.__class__.__name__
                    for inc in UNMANAGED_MODULES_INCLUDES
                ):
                    # unmanaged
                    module._memory_manager.unmanaged_modules.append(child_module)
                    modules_processed.append(child_module) # Avoid redundant processing
                else:
                    continue
