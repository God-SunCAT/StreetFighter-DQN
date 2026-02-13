import torch
import stable_retro
import cv2
from collections import deque
import numpy as np
import random

from network import LearningNet
from action_wrapper import SF2Discrete12

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
env = SF2Discrete12(env)

# 模型配置
weights = 'checkpoints/model_3000it.pkl'
net = LearningNet()
net.load_state_dict(torch.load(weights))
net.to('cuda')

# 环境操作
next_frame, info = env.reset()
state = deque(maxlen=4)
total_reward = 0
done = False
while not done:
    if len(state) >= 4:
        data = np.stack(list(state), axis=0)
        data = torch.tensor(data)
        data = data.unsqueeze(0) / 255
        data = data.float()
        data = data.to('cuda')
        result = net(data)
        action = int(torch.argmax(result, dim=-1)[0])
        print(action)
    else:
        # 最初的几步直接无动作
        action = 0

    next_frame, game_reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += game_reward
    
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
print(f'Total Reward: {total_reward}')

cv2.destroyAllWindows()
env.close()

