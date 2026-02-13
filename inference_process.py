import torch
from network import LearningNet
from replay_buffer import SharedReplayBuffer

def inference_worker(num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LearningNet().to(device)
    replay_buffers = [
        SharedReplayBuffer(
            capacity=5000,
            obs_shape=(4, 84, 84),
            name_prefix=f'rl_buffer_worker_{i}',
            create=False
        )
        for i in range(num_workers)
    ]

    while True:
        # 串行推理太慢了
        for buffer in replay_buffers:
            if buffer.get_caculate_state() == 1:
                state = torch.tensor(buffer.get_caculate_buffer()).to(device, dtype=torch.float32).div_(255.0).unsqueeze(0)
                
                with torch.no_grad():
                    buffer.get_caculate_state(value=False)[()] = torch.argmax(net(state), dim=-1)[0]
            
        
    