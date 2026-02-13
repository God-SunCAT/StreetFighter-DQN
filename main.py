import time
import multiprocessing as mp
from multiprocessing import shared_memory
import cv2
import numpy as np

from env_worker import worker
from monitor_grid import *

runing = True

if __name__ == "__main__":
    
    processes = []
    NUM_WORKERS = 20
    mp.set_start_method("spawn", force=True)
    
    queue = mp.Queue(maxsize=5000)
    grid_shape = (4, 5)
    frames_shape = (NUM_WORKERS, 200, 256, 3) # H, W, C
    nbytes = np.prod(frames_shape) * np.dtype(np.uint8).itemsize

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
        frames_shape,
        dtype=np.uint8,
        buffer=shm_latest_obs.buf
    )

    try:
        shm_tmp_int64 = shared_memory.SharedMemory(create=True, size=nbytes, name="tmp_int64")
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

    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(i, NUM_WORKERS, frames_shape, queue))
        p.start()
        processes.append(p)

    # === 训练循环 ===
    counter = 0
    time_stamp = 0
    while runing:
        image = create_monitor_grid(latest_obs, [str(i + 1) for i in range(NUM_WORKERS)], grid_shape[0], grid_shape[1])
        cv2.imshow(win_name, image)
        cv2.waitKey(1)

        if counter % 1000 == 0:
            div = time.time() - time_stamp
            if div > 0:
                # Gb
                total_bytes = tmp_int64 / 1024 / 1024 / 1024
                print(total_bytes / div, 'Gb/s')
                
                time_stamp = time.time()

        counter += 1
        # transition = queue.get()

    shm_latest_obs.close()
    shm_latest_obs.unlink()

    shm_tmp_int64.close()
    shm_tmp_int64.unlink()