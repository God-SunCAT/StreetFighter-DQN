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

        if obs_enemy_hp == 176 and obs_player_hp == 176:
            round_done = False

        # 双子星老师, 您发发力吧 /哭
        # 我自己都改了一天奖励机制了, 现在只能靠您了/呜呜呜
        # 双子星老师, 您太不给力了, 只能自己上了.
        if not round_done:
            # 不设置动作引导, 全部靠自学以避免骗奖励
            # 这里有个坑就是, 战败血量不会经过0而是直接 -1
            # 正向奖励倍数
            positive_coeff = 3.0
            if obs_player_hp < 0:
                # Loss
                reward = -math.pow(176, (obs_enemy_hp + 1) / (176 + 1))
                round_done = True
            elif obs_enemy_hp < 0:
                # Win
                reward = positive_coeff * math.pow(176, (obs_player_hp + 1) / 176 + 1)
                round_done = True
            else:
                # Fighting
                reward = positive_coeff * (current_enemy_health - obs_enemy_hp) \
                    - (current_health - obs_player_hp)
            
            # 规范化
            reward = reward * 0.001

            # 强制截断 (DQN 训练的最后一道防线)
            # 无论前面怎么算，单步奖励绝对不允许超过 [-1, 1]
            reward = max(min(reward, 1.0), -1.0)

        
        # 更新记录值供下一帧对比
        current_health = obs_player_hp
        current_enemy_health = obs_enemy_hp

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