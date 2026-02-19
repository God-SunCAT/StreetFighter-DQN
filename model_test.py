import math
import os
import torch
import stable_retro
import cv2
from collections import deque
import numpy as np
import argparse

from network import LearningNet
from action_wrapper import SF2Discrete15

# 终端参数
parser = argparse.ArgumentParser(description='模型测试工具')
parser.add_argument("--it", type=int, help="测试权重迭代次数")
parser.add_argument("--vec", action='store_true', help="显示权重")
parser.add_argument("--reward", action='store_true', help="显示奖励")

args = parser.parse_args()

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
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

if args.it is None:
    weights = 'inference_weights.pt'
    # 绿色加粗输出
    print(f"{GREEN}[SUCCESS]{RESET} 载入最新权重: {weights}")
else:
    weights = f'checkpoints/model_{args.it}it.pt'

    if os.path.exists(weights):
        print(f"{GREEN}[SUCCESS]{RESET} 载入 {GREEN}{args.it}it{RESET} 权重")
    else:
        print(f"{RED}[ERROR]{RESET} {args.it}it 权重文件不存在: {weights}")

# weights = 'checkpoints/model_400000it.pt'

# 载入网络
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

frame_skip_count = 0
frame_skip_action = 0

tmp_reward = 0

# skip -> 6 | stride -> 2
while not done:
    # 帧跳过
    skip: bool = frame_skip_count < 6
    if frame_skip_count < 6:
        if round_done:
            # Round Done 阶段跳过加载靠设 count = 0
            frame_skip_action = 0

        next_frame, _, terminated, truncated, info = env.step(frame_skip_action)
    else:
        if len(state) >= 4:
            data = np.stack(list(state), axis=0)
            data = torch.tensor(data)
            data = data.unsqueeze(0)
            data = data.float()
            data = data.to('cuda')
            result = net(data)
            action = int(torch.argmax(result, dim=-1)[0])
            if args.vec:
                print(result)
        else:
            # 最初的几步直接无动作
            action = 0

        if tmp_reward != 0 and args.reward:
            print(f'Reward: {tmp_reward:.3f}')

        tmp_reward = 0
        
        frame_skip_count = 0
        frame_skip_action = action

        next_frame, _, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    
    # display
    frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Street Fighter II", frame)
    # 必须要给 cv2 留出刷新时间，否则会白屏
    cv2.waitKey(5) 

    # ================
    # ---- REWARD ----
    # ================

    reward = 0
    obs_enemy_hp = info.get('enemy_health', 0)
    obs_player_hp = info.get('health', 0)

    if obs_enemy_hp == 176 and obs_player_hp == 176:
        current_health = 176
        current_enemy_health = 176
        round_done = False

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

    # 累计帧跳过产生的 reward
    tmp_reward += reward

    # ================
    # ---- REWARD ----
    # ================

    total_reward += reward

    # 压入状态
    if frame_skip_count % 2  == 0 or not skip:
        gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (84, 84))
        state.append(gray_frame)

    # 应该从1开始
    frame_skip_count += 1
    
    if round_done:
        # -1 代表角色死亡, 清空 state 重新开始
        state.clear()
        old_state = None
        frame_skip_count = -1

print(f'\033[1;44;37m Total Reward: {total_reward:.2f} \033[0m')

cv2.destroyAllWindows()
env.close()

