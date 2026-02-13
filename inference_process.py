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
        # for buffer in replay_buffers:
        #     if buffer.get_caculate_state() == 1:
        #         state = torch.tensor(buffer.get_caculate_buffer()).to(device, dtype=torch.float32).div_(255.0).unsqueeze(0)
                
        #         with torch.no_grad():
        #             buffer.get_caculate_state(value=False)[()] = torch.argmax(net(state), dim=-1)[0]

        # 并行推理
        # 1. 收集阶段
        active_indices = []
        states_to_infer = []

        for i, buffer in enumerate(replay_buffers):
            if buffer.get_caculate_state() == 1:
                # 预处理 state
                s = torch.tensor(buffer.get_caculate_buffer(), device=device, dtype=torch.float32).div_(255.0)
                # s 的形状应当是 (4, 84, 84)
                states_to_infer.append(s)
                active_indices.append(i)

        # 2. 并行推理阶段
        if states_to_infer:
            # 将 list 中的 tensor 堆叠为 (Batch, 4, 84, 84)
            batch_tensor = torch.stack(states_to_infer)
            
            with torch.no_grad():
                # 一次性推整个 Batch
                outputs = net(batch_tensor)  # 输出形状: (Batch, action_size)
                actions = torch.argmax(outputs, dim=-1)  # 输出形状: (Batch,)

            # 3. 结果分发阶段
            for idx, action in zip(active_indices, actions):
                # action 是一个零维标量 tensor，使用 .item() 存入 buffer
                replay_buffers[idx].get_caculate_state(value=False)[()] = action.item()
            
        
    