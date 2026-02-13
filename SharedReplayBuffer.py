import numpy as np
from multiprocessing import shared_memory

class SharedReplayBuffer:
    def __init__(self, capacity, obs_shape=(84, 84), name_prefix="rl_buffer", create=False):
        self.capacity = capacity
        self.obs_shape = obs_shape
        
        # 定义数据规格
        self.specs = {
            'obs': {'shape': (capacity, *obs_shape), 'dtype': np.uint8},
            'actions': {'shape': (capacity,), 'dtype': np.int64},
            'rewards': {'shape': (capacity,), 'dtype': np.float32},
            'next_obs': {'shape': (capacity, *obs_shape), 'dtype': np.uint8},
            'dones': {'shape': (capacity,), 'dtype': np.bool_},
            'cursor': {'shape': (), 'dtype': np.int64}
        }
        
        self.shms = {}
        self.data = {}

        for key, spec in self.specs.items():
            shm_name = f"{name_prefix}_{key}"
            if create:
                # 主进程创建
                nbytes = np.prod(spec['shape']) * np.dtype(spec['dtype']).itemsize
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=nbytes)
            else:
                # Worker/Learner 挂载
                shm = shared_memory.SharedMemory(name=shm_name)
            
            self.shms[key] = shm
            self.data[key] = np.ndarray(spec['shape'], dtype=spec['dtype'], buffer=shm.buf)

    def add(self, obs, action, reward, next_obs,done):
        """Worker 调用：添加经验"""
        idx = self.data['cursor'][()]
        
        self.data['obs'][idx] = obs
        self.data['actions'][idx] = action
        self.data['rewards'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['dones'][idx] = done
        
        # 循环覆盖
        self.data['cursor'][()] = (idx + 1) % self.capacity

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