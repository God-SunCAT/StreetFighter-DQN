import math
import torch
import stable_retro
import cv2
from collections import deque
import numpy as np
import random

from network import LearningNet
from action_wrapper import SF2Discrete15

# 显示配置
win_name = "Street Fighter II"
cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL) # 使用简洁模式，干掉状态栏和工具栏
cv2.resizeWindow(win_name, 640, 480)

# 环境配置
env = stable_retro.make(
    game="StreetFighterIISpecialChampionEdition-Genesis-v0",
    state="Champion.Level12.RyuVsBison",
    render_mode='rgb_array'
)
env = SF2Discrete15(env)

# 模型配置
weights = 'inference_weights.pt'
# weights = 'checkpoints/model_18000it.pt'
net = LearningNet()
net.load_state_dict(torch.load(weights))
net.to('cuda')

# 环境操作
next_frame, info = env.reset()

state = deque(maxlen=4)
total_reward = 0
repeat_action = 0
repeat_count = 0
done = False

current_health = 176
current_enemy_health = 176
round_done = False

while not done:
    if len(state) >= 4:
        data = np.stack(list(state), axis=0)
        data = torch.tensor(data)
        data = data.unsqueeze(0)
        data = data.float()
        data = data.to('cuda')
        result = net(data)
        print(result)
        action = int(torch.argmax(result, dim=-1)[0])
    else:
        # 最初的几步直接无动作
        action = 0

    # action = env.action_space.sample()
    
    next_frame, game_reward, terminated, truncated, info = env.step(action)
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
        if round_done and delta_enemy_hp != 0 and delta_player_hp != 0:
            # 青色 加粗
            print('\033[1;36m' + '---- Next Round ----' + '\033[0m')
        
        round_done = False

    if obs_enemy_hp == -1 or obs_player_hp == -1:
        # -1 代表角色死亡, 防止二次奖励
        round_done = True

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
    
    if round_done:
        # -1 代表角色死亡, 清空 state 重新开始
        state.clear()
        old_state = None

    # 更新记录值供下一帧对比
    current_health = obs_player_hp
    current_enemy_health = obs_enemy_hp

    # if reward != 0:
    #     print(f'Reward: {reward}, Health: {current_health} (Δ {delta_player_hp}), EnemyHealth: {current_enemy_health} (Δ {obs_enemy_hp})')

    if abs(reward) > 0.1:
        # 定义临时颜色变量
        C_RWD = '\033[92m' if reward > 0 else '\033[91m' # 奖励正绿负红
        C_D_P = '\033[91m' if delta_player_hp > 0 else '' # 自己掉血显红
        C_D_E = '\033[92m' if delta_enemy_hp > 0 else '' # 敌人掉血显绿
        C_NUM = '\033[93m' # 数值亮黄
        C_END = '\033[0m'

        print(f'{C_RWD}Reward: {reward:.4f}{C_END}, '
            f'Health: {C_NUM}{current_health}{C_END} ({C_D_P}Δ {delta_player_hp}{C_END}), '
            f'EnemyHealth: {C_NUM}{current_enemy_health}{C_END} ({C_D_E}Δ {delta_enemy_hp}{C_END})')
        
    # ================
    # ---- REWARD ----
    # ================
    
    total_reward += reward
    
    # 压入状态
    gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (84, 84))
    state.append(gray_frame)

    # 显示
    frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Street Fighter II", frame)
    # 必须要给 cv2 留出刷新时间，否则会白屏
    cv2.waitKey(5)

print(f'\033[1;44;37m Total Reward: {total_reward:.2f} \033[0m')

cv2.destroyAllWindows()
env.close()

