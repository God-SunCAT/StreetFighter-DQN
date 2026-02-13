import numpy as np
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