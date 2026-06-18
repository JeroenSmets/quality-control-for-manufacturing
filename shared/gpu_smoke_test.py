import torch
import time

print(torch.cuda.is_available())  # <backend_name> is cuda, mps, or xpu

print(f'device name [0]:', torch.cuda.get_device_name(0))

print(torch.utils.collect_env)

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor1 = torch.randn(1000000000)
tensor2 = torch.randn(1000000000)

tensor1.to(device)
tensor2.to(device)

# Initialize events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record
start_event.record()
result = tensor1 * tensor2 # Perform operation
end_event.record()

# Synchronize to wait for GPU to finish
torch.cuda.synchronize()

# Time in milliseconds
print(start_event.elapsed_time(end_event))