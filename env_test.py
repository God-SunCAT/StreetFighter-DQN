import torch
import stable_retro
import cv2
from collections import deque
import numpy as np

from network import LearningNet
from action_wrapper import SF2Discrete15

# ChatGPT

# ================= 配置 =================
win_name = "Street Fighter II - Manual Test Mode"
cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(win_name, 640, 480)

# 环境初始化
env = stable_retro.make(
    game="StreetFighterIISpecialChampionEdition-Genesis-v0",
    state="Champion.Level12.RyuVsBison",
    render_mode='rgb_array'
)
env = SF2Discrete15(env)

# 键盘映射表 (OpenCV KeyCode -> Action ID)
# 主键盘 1-9 -> Action 1-9
KEY_MAP = {ord(str(i)): i for i in range(1, 10)}

# 主键盘 0-7 -> Action 10-17
for i in range(8):
    KEY_MAP[ord(str(i))] = 10 + i

# 状态追踪
next_frame, info = env.reset()
current_health = 176
current_enemy_health = 176
total_reward = 0
done = False
round_done = False

print("\n" + "="*30)
print("  街霸2 手动测试模式启动")
print("  控制说明:")
print("  - 主键盘 1-9: 动作 1-9")
print("  - 主键盘 0: 动作 10 (中拳)")
print("  - 主键盘 1: 动作 11 (重拳)")
print("  - 主键盘 2: 动作 12 (轻脚)")
print("  - 主键盘 3: 动作 13 (中脚)")
print("  - 主键盘 4: 动作 14 (重脚)")
print("  - 主键盘 5: 动作 15 (前插重拳)")
print("  - 主键盘 6: 动作 16 (扫堂腿)")
print("  - 主键盘 7: 动作 17 (三拳同按)")
print("  - ESC: 退出游戏")
print("  - 无输入时: 自动执行 Action 0")
print("="*30 + "\n")

while not done:
    # --- 1. 获取键盘输入 ---
    # waitKey(10) 提供约 100FPS 的刷新率，适合手动操作
    key = cv2.waitKey(100) & 0xFFFF
    
    if key == 27: # ESC 退出
        break
    
    # 映射动作，默认 0
    action = KEY_MAP.get(key, 0)

    # --- 2. 环境步进 ---
    # 为了手感，每个按键动作持续执行 4 帧 (Frame Skip)
    for _ in range(4):
        next_frame, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
            break

    # --- 3. 奖励逻辑计算 (保留你的逻辑用于观察) ---
    reward = 0
    obs_enemy_hp = info.get('enemy_health', 0)
    obs_player_hp = info.get('health', 0)
    delta_enemy_hp = obs_enemy_hp - current_enemy_health
    delta_player_hp = obs_player_hp - current_health

    # 检测新回合
    if obs_enemy_hp == 176 or obs_player_hp == 176:
        if round_done and delta_enemy_hp != 0 and delta_player_hp != 0:
            print('\033[1;36m' + '---- Next Round ----' + '\033[0m')
        round_done = False

    if obs_enemy_hp <= 0 or obs_player_hp <= 0:
        round_done = True

    if not round_done:
        if delta_enemy_hp < 0:
            reward += 12 ** ((176 - obs_enemy_hp) / 176) * 3 + 0.5
        if delta_player_hp < 0:
            reward -= 10 ** ((obs_enemy_hp / 176) - 0.1)
    
    current_health, current_enemy_health = obs_player_hp, obs_enemy_hp
    reward *= 0.1
    total_reward += reward

    # --- 4. 打印反馈 ---
    if action != 0 or reward != 0:
        C_RWD = '\033[92m' if reward > 0 else ('\033[91m' if reward < 0 else '')
        C_NUM = '\033[93m'
        C_END = '\033[0m'
        print(f"Action: {action} | {C_RWD}Reward: {reward:.2f}{C_END} | HP: {C_NUM}{current_health}{C_END} | E-HP: {C_NUM}{current_enemy_health}{C_END}")

    # --- 5. 渲染画面 ---
    frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))
    
    # 在画面上实时显示当前 Action
    cv2.putText(frame, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(win_name, frame)

print(f'\n\033[1;44;37m Final Total Reward: {total_reward:.2f} \033[0m')
cv2.destroyAllWindows()
env.close()