import stable_retro
import cv2
import numpy as np
from multiprocessing import shared_memory

def worker(worker_id, num_workers, frames_shape, queue):
    shm_latest_obs = shared_memory.SharedMemory(name="latest_obs")
    latest_obs = np.ndarray(
        frames_shape,
        dtype=np.uint8,
        buffer=shm_latest_obs.buf
    )

    shm_tmp_int64 = shared_memory.SharedMemory(name="tmp_int64")
    tmp_int64 = np.ndarray(
        (),
        dtype=np.int64,
        buffer=shm_tmp_int64.buf
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
        
        # display
        if np.array_equal(latest_obs[worker_id], next_obs):
            stable_count += 1
        else:
            stable_count = 0
            latest_obs[worker_id] = next_obs

        frame = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (84, 84))
        tmp_int64[()] += frame.nbytes
        
        obs = next_obs
        if done or stable_count >= 30:
            # 30次不动是最合适的参数, 自动跳失败和通关
            obs, info = env.reset()

    shm_latest_obs.close()
    shm_tmp_int64.close()
