import time
import stable_retro
import cv2
import math
import numpy as np
from collections import deque

from multiprocessing import shared_memory

from replay_buffer import SharedReplayBuffer
from action_wrapper import SF2Discrete15


def env_worker(worker_id, num_workers):
    # 显示配置
    shm_latest_obs = shared_memory.SharedMemory(name="latest_obs")
    latest_obs = np.ndarray(
        (num_workers, 200, 256, 3),
        dtype=np.uint8,
        buffer=shm_latest_obs.buf
    )

    shm_tmp_int64 = shared_memory.SharedMemory(name="tmp_int64")
    tmp_int64 = np.ndarray(
        (),
        dtype=np.int64,
        buffer=shm_tmp_int64.buf
    )
    # ReplayBuffer
    replay_buffer = SharedReplayBuffer(
        capacity=5000,
        obs_shape=(4, 84, 84),
        name_prefix=f'rl_buffer_worker_{worker_id}',
        create=False
    )

    # 环境配置
    env = stable_retro.make(
        game="StreetFighterIISpecialChampionEdition-Genesis-v0",
        state="Champion.Level12.RyuVsBison",
        render_mode='rgb_array'
    )
    env = SF2Discrete15(env)

    # 环境操作
    next_frame, info = env.reset()

    # ---- REWARD ----

    current_health = 176
    current_enemy_health = 176
    round_done = False

    # ---- REWARD ----

    state = deque(maxlen=4)
    old_state = None # (state, action, reward)
    stable_count = 0
    total_reward = 0

    time_stamp = time.time()
    while True:
        if len(state) >= 4:
            replay_buffer.get_caculate_buffer()[:] = state
            # 1号为要求处理, 其他为处理完毕
            replay_buffer.get_caculate_state(value=False)[()] = 1
            while True:
                # 等待处理返回
                if replay_buffer.get_caculate_state() != 1:
                    action = int(replay_buffer.get_caculate_state()) - 3 # 3是偏移量
                    # 设置为无任务状态
                    replay_buffer.get_caculate_state(value=False)[()] = 0
                    break
        else:
            # 最初的几步直接无动作
            action = 0
            
        next_frame, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # ================
        # ---- REWARD ----
        # ================

        reward = 0
        obs_enemy_hp = info.get('enemy_health', 0)
        obs_player_hp = info.get('health', 0)

        delta_enemy_hp = obs_enemy_hp - current_enemy_health
        delta_player_hp = obs_player_hp - current_health

        if obs_enemy_hp == 176 or obs_player_hp == 176:
            round_done = False

        if obs_enemy_hp == -1 or obs_player_hp == -1:
            # -1 代表角色死亡, 防止二次奖励
            round_done = True

        if not round_done:
            if action <= 8 and action >= 0:
                # 中性动作惩罚
                reward -= 0.06
            if action <= 14 and action >= 9:
                # 攻击动作奖励
                reward += 0.05

            if delta_enemy_hp < 0:
                # 攻击时敌人血量越少, 奖励越大
                # print('EnemyDelta', delta_enemy_hp)
                reward += 12 ** ((176 - obs_enemy_hp) / 176) * 3 + 0.5

            if delta_player_hp < 0:
                # 受击时候敌人血量越少, 惩罚越大
                # print('PlayerDelta', delta_player_hp)
                reward -= 12 ** (((176 - obs_enemy_hp) / 176) - 0.1) + 1
        
        # 更新记录值供下一帧对比
        current_health = obs_player_hp
        current_enemy_health = obs_enemy_hp

        # 奖励归一化：将奖励缩放到神经网络易于处理的范围
        reward = reward * 0.1

        # ================
        # ---- REWARD ----
        # ================

        total_reward += reward

        # display
        if np.array_equal(latest_obs[worker_id], next_frame):
            stable_count += 1
        else:
            stable_count = 0
            latest_obs[worker_id] = next_frame

        # 压入状态
        gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (84, 84))
        state.append(gray_frame)

        # 压入缓冲区
        if len(state) >= 4 and not round_done:
            if old_state is not None:
                replay_buffer.add(
                    obs=old_state[0],
                    action=old_state[1],
                    reward=old_state[2],
                    next_obs=np.stack(list(state), axis=0),
                    done=False
                )

            # 处理 Done 状态
            if obs_enemy_hp == 0 or obs_player_hp == 0:
                tmp_obs = np.stack(list(state), axis=0)
                replay_buffer.add(
                    obs=tmp_obs,
                    action=action,
                    reward=reward,
                    next_obs=tmp_obs, # 真正跑的时候这里是什么数据都没区别
                    done=True
                )
            
            old_state = (np.stack(list(state), axis=0), action, reward)

        if round_done:
            # -1 代表角色死亡, 清空 state 重新开始
            state.clear()
            old_state = None

        # 数据量统计(只统计得到的画面数据量)
        tmp_int64[()] += gray_frame.nbytes
        
        if done or stable_count >= 30:
            # 30次不动是最合适的参数, 自动跳失败和通关
            
            replay_buffer.data['statistic_time'][()] += time.time() - time_stamp
            replay_buffer.data['statistic_reward'][()] += total_reward
            replay_buffer.data['statistic_count'][()] += 1
            time_stamp = time.time()

            obs, info = env.reset()
            
            total_reward = 0

            # ---- REWARD ----
            
            current_health = 176
            current_enemy_health = 176
            round_done = False

            # ---- REWARD ----

    shm_latest_obs.close()
    shm_tmp_int64.close()

    replay_buffer.close()