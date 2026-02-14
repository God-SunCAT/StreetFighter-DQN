import torch
import numpy as np
from multiprocessing import shared_memory

from network import LearningNet
from replay_buffer import SharedReplayBuffer

def inference_worker(num_workers):
    # 这里其实没必要兼容 cpu, cuda 跑不了的话, 按照 cpu 的速度来说, 也没有跑的必要了
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = LearningNet().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    shm_inference_net_version = shared_memory.SharedMemory(name="inference_net_version")
    inference_net_version = np.ndarray((), dtype=np.int64, buffer=shm_inference_net_version.buf)

    shm_episilon = shared_memory.SharedMemory(name="rl_episilon")
    episilon = np.ndarray((), dtype=np.float64, buffer=shm_episilon.buf)

    replay_buffers = [
        SharedReplayBuffer(
            capacity=100000,
            obs_shape=(4, 84, 84),
            name_prefix=f'rl_buffer_worker_{i}',
            create=False
        )
        for i in range(num_workers)
    ]
    current_vesion = 0
    while True:
        # 并行推理
        # 1. 收集阶段
        active_indices = []
        states_to_infer = []

        for i, buffer in enumerate(replay_buffers):
            if buffer.get_caculate_state() == 1:
                if np.random.rand() < episilon[()].item():
                    action = int(np.random.randint(0, 17))
                    buffer.get_caculate_state(value=False)[()] = action + 3
                    continue

                # 预处理 state
                s = torch.tensor(buffer.get_caculate_buffer(), device=device, dtype=torch.float32)
                # s 的形状应当是 (4, 84, 84)
                states_to_infer.append(s)
                active_indices.append(i)

        # 2. 并行推理阶段
        if states_to_infer:
            # 将 list 中的 tensor 堆叠为 (Batch, 4, 84, 84)
            batch_tensor = torch.stack(states_to_infer)
            
            # 检测版本
            if inference_net_version[()].item() != current_vesion:
                state_dict = torch.load(
                    "inference_weights.pt",
                    map_location="cpu"   # 关键
                )
                current_vesion = inference_net_version[()].item()
                
            with torch.no_grad():
                # 一次性推整个 Batch
                outputs = net(batch_tensor)  # 输出形状: (Batch, action_size)
                actions = torch.argmax(outputs, dim=-1)  # 输出形状: (Batch,)

            # 3. 结果分发阶段
            for idx, action in zip(active_indices, actions):
                # action 是一个零维标量 tensor，使用 .item() 存入 buffer
                replay_buffers[idx].get_caculate_state(value=False)[()] = action.item() + 3

    shm_inference_net_version.close()
    shm_episilon.close()
            
        
    