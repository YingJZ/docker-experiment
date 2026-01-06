import torch
from safetensors.torch import save_file

# 1. 加载旧模型
model = torch.load("batch/mobilenet_v2-imagenet1k-v1.pth", map_location="cpu")
# 如果是 state_dict，直接用；如果是完整模型对象，取 model.state_dict()
if hasattr(model, "state_dict"):
    state_dict = model.state_dict()
else:
    state_dict = model

# 2. 保存为 safetensors 格式
save_file(state_dict, "batch/mobilenet_v2-imagenet1k-v1.safetensors")
