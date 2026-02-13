import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from replay_buffer import SharedReplayBuffer
from network import LearningNet
def train_worker(num_workers):
    # 加载缓冲区
    replay_buffers = [
        SharedReplayBuffer(
            capacity=5000,
            obs_shape=(4, 84, 84),
            name_prefix=f'rl_buffer_worker_{i}',
            create=False
        )
        for i in range(num_workers)
    ]

    # 等待缓冲区填充
    print('<等待第一轮缓冲>')
    wait_count = 0
    while wait_count != num_workers:
        wait_count = 0
        for buffer in replay_buffers:
            if buffer.get_state():
                wait_count += 1
    print('<第一轮缓冲完毕>')
    # 训练循环
    # while True:
    #     pass
                
        