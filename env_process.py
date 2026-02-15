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

        if obs_enemy_hp == 176 and obs_player_hp == 176:
            round_done = False

        if obs_enemy_hp == -1 or obs_player_hp == -1:
            # -1 代表角色死亡, 防止二次奖励
            round_done = True

        # 双子星老师, 您发发力吧 /哭
        # 我自己都改了一天奖励机制了, 现在只能靠您了/呜呜呜
        if not round_done:
            # 1. 基础引导：大幅降低量级，防止背景惩罚盖过攻击奖励
            if 0 <= action <= 8:
                reward -= 0.001  # 只是微小的惩罚，防止原地挂机
            elif 9 <= action <= 14:
                reward += 0.01   # 鼓励出招

            # 计算血量进度 (0.0 ~ 1.0)
            progress = (176 - obs_enemy_hp) / 176

            # 2. 攻击奖励：将底数从 12 降到 2，确保单步奖励不会过载
            if delta_enemy_hp < 0:
                # 打中瞬间：基础奖励 0.3 + 进度加成 (最高 0.6) = 0.9
                reward += (2 ** progress) * 0.3 + 0.3

            # 3. 受击惩罚：同样限制在 -1 附近
            if delta_player_hp < 0:
                # 被打瞬间：基础惩罚 -0.4 + 进度加成 (最高 -0.4) = -0.8
                reward -= (2 ** (progress - 0.1)) * 0.4 + 0.1

            # 4. 强制截断 (DQN 训练的最后一道防线)
            # 无论前面怎么算，单步奖励绝对不允许超过 [-2, 2]
            reward = max(min(reward, 2.0), -2.0)

            # 注意：这里不再需要 reward * 0.1 了，
            # 因为我们在上面已经手动把数值精确控制在 [-1, 1] 附近的黄金区间。
        
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