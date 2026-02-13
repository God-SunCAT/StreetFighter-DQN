import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import shared_memory
import copy

from replay_buffer import SharedReplayBuffer, sample_and_merge
from network import LearningNet

def train_worker(num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载缓冲区
    shm_inference_net_version = shared_memory.SharedMemory(name="inference_net_version")
    inference_net_version = np.ndarray((), dtype=np.int64, buffer=shm_inference_net_version.buf)
    
    shm_episilon = shared_memory.SharedMemory(name="rl_episilon")
    episilon = np.ndarray((), dtype=np.float64, buffer=shm_episilon.buf)

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
    
    # 训练循环 -> 从这里开始, 完全允许自定义
    # 注意迁移网络时候需要设状态
    # 允许 net 直接加载文件中的 checkpoint
    net = LearningNet().to(device)
    net_target = copy.deepcopy(net)
    for p in net_target.parameters():
        p.requires_grad_(False)

    # 训练参数
    EPS_START = 1.0    # 初始探索概率（全乱走）
    EPS_END = 0.05     # 最终探索概率（保留一小部分随机性）
    EPS_DECAY = 50000  # 衰减率参数

    gamma = 0.99
    episilon[()] = 1 # 按迭代次数衰减
    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr=1e-4,              # 学习率
        betas=(0.9, 0.999),   # 动量参数
        eps=1e-8,             # 防止除以 0 的微小值
        weight_decay=0.01,    # 权重衰减系数（AdamW 的核心）
        amsgrad=False         # 是否使用 AMSGrad 变体
    )

    it_count = 0
    trans_count = {'infer': 0, 'target': 0}
    while True:
        # 构建样本
        # 20 个采集, 选4个缓存池, 每个取 48/4 = 12个
        # 除法结果必须是整数, 这里是整除不代表允许忽略小数
        batch = 48
        select_buffer = 4
        batch_per_buffer = batch // select_buffer

        buffer_list = np.random.choice(20, select_buffer, replace=False)
        batch_obs, batch_act, batch_rew, batch_next_obs, batch_done = \
            sample_and_merge(
                buffer_list=[replay_buffers[i] for i in buffer_list],
                batch_size_per_buffer=batch_per_buffer,
                device='cuda'
            )

        qs = net(batch_obs)
        q = qs[range(batch), batch_act]

        # pytorch 遵循设备一致性原则, next_q 与 next_qs 在一个设备上
        next_qs = net_target(batch_next_obs)
        next_q = torch.max(next_qs, dim=1).values

        target = batch_rew + (~batch_done).float() * gamma * next_q

        loss = F.mse_loss(q, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 推理网络迁移
        if it_count % 50 == 0:
            # 我真没招了, 直接掉文件系统做跨进程权重同步吧
            torch.save(net.state_dict(), "inference_weights.pt")
            trans_count['infer'] += 1
            inference_net_version[()] = trans_count['infer']

        # 目标网络迁移
        if it_count % 5000 == 0:
            for main_param, target_param in zip(net.parameters(), net_target.parameters()):
                target_param.copy_(main_param)
            trans_count['target'] += 1
            print(f'迭代次数: {it_count} | 推理迁移次数: {trans_count['infer']} | 目标迁移次数: {trans_count['target']} | 探索概率: {episilon[()].item():.3f}')
            
        if it_count % 3000 == 0:
            torch.save(net.state_dict(), f'./checkpoints/model_{it_count}it.pkl')
            print(f'<权重已保存>迭代次数: {it_count} | 推理迁移次数: {trans_count['infer']} | 目标迁移次数: {trans_count['target']} | 探索概率: {episilon[()].item():.3f}')
                
        if it_count % 100 == 0:
            print(f'迭代次数: {it_count} | 推理迁移次数: {trans_count['infer']} | 目标迁移次数: {trans_count['target']} | 探索概率: {episilon[()].item():.3f}')
                
        if os.path.exists('./STOP'):
            torch.save(net.state_dict(), f'./checkpoints/model_{it_count}it.pkl')
            print(f'<权重已保存>迭代次数: {it_count} | 推理迁移次数: {trans_count['infer']} | 目标迁移次数: {trans_count['target']} | 探索概率: {episilon[()].item():.3f}')
            print(f'<训练中止>')
            break
        
        episilon[()] = EPS_END + (EPS_START - EPS_END) * \
           math.exp(-1. * it_count / EPS_DECAY)

        it_count += 1

    # 训练循环结束, 不允许自定义            
    shm_inference_net_version.close()
    shm_episilon.close()