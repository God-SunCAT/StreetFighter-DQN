import torch
from network import LearningNet
def load_saved_net(device='cuda') -> tuple[LearningNet, int]:
    saved_it = 0
    weight_path = ''

    net: LearningNet = LearningNet().to(device)
    if saved_it:
        state_dict = torch.load(
            weight_path,
            map_location="cpu"   # 关键
        )
        net.load_state_dict(state_dict)
    return net, saved_it