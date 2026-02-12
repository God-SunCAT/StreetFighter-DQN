import stable_retro
import cv2
import numpy as np
from multiprocessing import shared_memory

def worker(worker_id, num_workers, frames_shape, queue):
    shm = shared_memory.SharedMemory(name="latest_obs")
    latest_obs = np.ndarray(
        frames_shape,
        dtype=np.uint8,
        buffer=shm.buf
    )

    env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0",
        state="Champion.Level12.RyuVsBison",
        render_mode='rgb_array'
    )

    obs, info = env.reset()
    stable_count = 0
    while True:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        
        if np.array_equal(latest_obs[worker_id], next_obs):
            stable_count += 1
        else:
            stable_count = 0
            latest_obs[worker_id] = next_obs
        # queue.put((obs, action, reward, next_obs, done))

        obs = next_obs
        if done or stable_count >= 30:
            # 30次不动是最合适的参数, 自动跳失败和通关
            obs, info = env.reset()

