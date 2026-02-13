import numpy as np
import torch
from multiprocessing import shared_memory

class SharedReplayBuffer:
    def __init__(self, capacity, obs_shape=(4, 84, 84), name_prefix="rl_buffer", create=False):
        self.capacity = capacity
        self.obs_shape = obs_shape
        
        # 定义数据规格
        self.specs = {
            'obs': {'shape': (capacity, *obs_shape), 'dtype': np.uint8},
            'actions': {'shape': (capacity,), 'dtype': np.int64},
            'rewards': {'shape': (capacity,), 'dtype': np.float32},
            'next_obs': {'shape': (capacity, *obs_shape), 'dtype': np.uint8},
            'dones': {'shape': (capacity,), 'dtype': np.bool_},
            'cursor': {'shape': (), 'dtype': np.int64},

            'caculate_state': {'shape': (), 'dtype': np.uint8}, # 0 无计算任务 | 1 需要计算 | 2 计算完成
            'caculate_buffer': {'shape': obs_shape, 'dtype': np.uint8},

            'buffer_state': {'shape': (), 'dtype': np.uint8} # 整个缓冲区的状态
        }
        
        self.shms = {}
        self.data = {}

        for key, spec in self.specs.items():
            shm_name = f"{name_prefix}_{key}"
            if create:
                nbytes = int(np.prod(spec['shape']) * np.dtype(spec['dtype']).itemsize)
                try:
                    # 主进程创建
                    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
                except FileExistsError:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                    shm.unlink()
                    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)
            else:
                # Worker/Learner 挂载
                shm = shared_memory.SharedMemory(name=shm_name)
            
            self.shms[key] = shm
            self.data[key] = np.ndarray(spec['shape'], dtype=spec['dtype'], buffer=shm.buf)

    def get_state(self):
        return self.data['buffer_state'][()]
    
    def get_caculate_state(self, value=True):
        if value:
            return self.data['caculate_state'][()]
        return self.data['caculate_state']
    
    def get_caculate_buffer(self):
        return self.data['caculate_buffer']

    def add(self, obs, action, reward, next_obs, done):
        """Worker 调用：添加经验"""

        idx = self.data['cursor'][()]
        
        self.data['obs'][idx] = obs
        self.data['actions'][idx] = action
        self.data['rewards'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['dones'][idx] = done
        
        # 循环覆盖
        self.data['cursor'][()] = (idx + 1) % self.capacity

        if (idx + 1) % self.capacity == 0:
            # 缓冲区填满, 可以开始训练
            self.data['buffer_state'][()] = 1

    def sample(self, batch_size):
        """Learner 调用：随机采样索引"""
        # 早期等写满几千条再开始采
        upper_bound = self.capacity
        indices = np.random.randint(0, upper_bound, size=batch_size)
        
        # 返回的是 NumPy 视图（零拷贝）
        return (
            self.data['obs'][indices],
            self.data['actions'][indices],
            self.data['rewards'][indices],
            self.data['next_obs'][indices],
            self.data['dones'][indices]
        )

    def close(self):
        for shm in self.shms.values():
            shm.close()

    def unlink(self):
        for shm in self.shms.values():
            shm.unlink()

def sample_and_merge(buffer_list, batch_size_per_buffer, device):
    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

    # 1. 遍历 list，每个对象采样出一小块 NumPy 数组
    for buf in buffer_list:
        o, a, r, no, d = buf.sample(batch_size_per_buffer)
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        next_obs_list.append(no)
        done_list.append(d)

    # 2. 在 CPU 上进行 NumPy 拼接 (非常快)
    # np.concatenate 比 torch.cat 在处理大量小数组时更丝滑
    all_obs = np.concatenate(obs_list, axis=0)
    all_act = np.concatenate(act_list, axis=0)
    all_rew = np.concatenate(rew_list, axis=0)
    all_next_obs = np.concatenate(next_obs_list, axis=0)
    all_done = np.concatenate(done_list, axis=0)

    # 3. 一次性转为 Tensor 并推送到 GPU
    # 使用 torch.as_tensor 或 from_numpy 避免不必要的内存拷贝
    # 然后再调用 .to(device, non_blocking=True) 开启异步传输
    batch_obs = torch.as_tensor(all_obs, device=device, dtype=torch.float32)
    batch_act = torch.as_tensor(all_act, device=device, dtype=torch.int64)
    batch_rew = torch.as_tensor(all_rew, device=device, dtype=torch.float32)
    batch_next_obs = torch.as_tensor(all_next_obs, device=device, dtype=torch.float32)
    batch_done = torch.as_tensor(all_done, device=device, dtype=torch.bool)

    return batch_obs, batch_act, batch_rew, batch_next_obs, batch_done