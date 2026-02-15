import torch
from network import LearningNet
def load_saved_net(device='cuda') -> tuple[LearningNet, int]:
    weight_path = 'model_566301it.pt'
    saved_it = 566301
    state_dict = torch.load(
        weight_path,
        map_location="cpu"   # 关键
    )
    net: LearningNet = LearningNet().to(device)
    net.load_state_dict(state_dict)
    return net, saved_it