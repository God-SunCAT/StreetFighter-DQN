import time
import multiprocessing as mp
from multiprocessing import shared_memory
import cv2
import numpy as np

from env_process import env_worker
from train_process import train_worker
from inference_process import inference_worker

from replay_buffer import SharedReplayBuffer
from monitor_grid import *
from network import LearningNet

runing = True

if __name__ == "__main__":

    # 多线程
    processes = []
    train_process = None
    inference_process = None
    NUM_WORKERS = 20
    mp.set_start_method("spawn", force=True)
    
    # 显示配置
    grid_shape = (4, 5)
    nbytes = np.prod((NUM_WORKERS, 200, 256, 3)) * np.dtype(np.uint8).itemsize

    try:
        # 尝试创建
        shm_latest_obs = shared_memory.SharedMemory(create=True, size=nbytes, name="latest_obs")
    except FileExistsError:
        # 如果已存在，先清理再创建
        shm_latest_obs = shared_memory.SharedMemory(name="latest_obs")
        shm_latest_obs.close()
        shm_latest_obs.unlink()
        shm_latest_obs = shared_memory.SharedMemory(create=True, size=nbytes, name="latest_obs")

    latest_obs = np.ndarray(
        (NUM_WORKERS, 200, 256, 3),
        dtype=np.uint8,
        buffer=shm_latest_obs.buf
    )

    try:
        shm_tmp_int64 = shared_memory.SharedMemory(create=True, size=8, name="tmp_int64")
    except FileExistsError:
        shm_tmp_int64 = shared_memory.SharedMemory(name="tmp_int64")
        shm_tmp_int64.close()
        shm_tmp_int64.unlink()
        shm_tmp_int64 = shared_memory.SharedMemory(create=True, size=8, name="tmp_int64")

    tmp_int64 = np.ndarray(
        (),
        dtype=np.int64,
        buffer=shm_tmp_int64.buf
    )

    win_name = "Street Fighter II"
    win_shape = get_grid_final_shape(
        num_workers=NUM_WORKERS,
        rows=grid_shape[0],
        cols=grid_shape[1]
    )
    
    scaling = 1.4
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win_name, int(win_shape[0] * scaling), int(win_shape[1] * scaling))

    # 推理用网络(目标网络 -> 仅支持GPU) -> GPU天生支持共享权重
    # 8byte -> np.int64
    try:
        shm_inference_net_version = shared_memory.SharedMemory(create=True, size=8, name="inference_net_version")
    except FileExistsError:
        shm_inference_net_version = shared_memory.SharedMemory(name="inference_net_version")
        shm_inference_net_version.close()
        shm_inference_net_version.unlink()
        shm_inference_net_version = shared_memory.SharedMemory(create=True, size=8, name="inference_net_version")

    # 8byte -> float64
    try:
        shm_episilon = shared_memory.SharedMemory(create=True, size=8, name="rl_episilon")
    except FileExistsError:
        shm_episilon = shared_memory.SharedMemory(name="rl_episilon")
        shm_episilon.close()
        shm_episilon.unlink()
        shm_episilon = shared_memory.SharedMemory(create=True, size=8, name="rl_episilon")
    
    episilon = np.ndarray(
        (),
        dtype=np.float64,
        buffer=shm_episilon.buf
    )

    episilon[()] = 1
    # 启动推理线程
    inference_process = mp.Process(target=inference_worker, args=(NUM_WORKERS,))
    inference_process.start()

    # 启动训练线程
    replay_buffers = [
        SharedReplayBuffer(
            capacity=5000,
            obs_shape=(4, 84, 84),
            name_prefix=f'rl_buffer_worker_{i}',
            create=True
        )
        for i in range(NUM_WORKERS)
    ]
    train_process = mp.Process(target=train_worker, args=(NUM_WORKERS,))
    train_process.start()
    
    # 启动环境线程
    for i in range(NUM_WORKERS):
        p = mp.Process(target=env_worker, args=(i, NUM_WORKERS))
        p.start()
        processes.append(p)
    
    # 显示循环
    counter = 0
    time_stamp = 0
    while runing:
        image = create_monitor_grid(latest_obs, [str(i + 1) for i in range(NUM_WORKERS)], grid_shape[0], grid_shape[1])
        cv2.imshow(win_name, image)
        cv2.waitKey(1)

        if counter % 1000 == 0:
                print(f'累计 {tmp_int64 / 1024 / 1024 / 1024:.3f} Gb')
                time_stamp = time.time()

        counter += 1

    shm_latest_obs.close()
    shm_latest_obs.unlink()

    shm_tmp_int64.close()
    shm_tmp_int64.unlink()

    shm_inference_net_version.close()
    shm_inference_net_version.unlink()

    shm_episilon.close()
    shm_episilon.unlink()

    for buffer in replay_buffers:
        buffer.close()
        buffer.unlink()