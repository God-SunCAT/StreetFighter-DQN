import numpy as np
import gymnasium as gym

class SF2Discrete12(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        
        actions = [
            [],                     # 0: 不动
            ['LEFT'],               # 1: 后退
            ['RIGHT'],              # 2: 前进 (脱离墙角关键)
            ['UP'],                 # 3: 跳
            ['DOWN'],               # 4: 下蹲
            ['DOWN', 'LEFT'],       # 5: 蹲防
            ['DOWN', 'RIGHT'],      # 6: 斜下 (进攻过渡/滑铲，防缩墙角核心)
            ['UP', 'LEFT'],         # 7: 后跳
            ['UP', 'RIGHT'],        # 8: 前跳 (跳出包围圈)
            # 攻击键 (合并最常用的几个)
            ['A'],                  # 9: 轻拳
            ['B'],                  # 10: 中拳
            ['C'],                  # 11: 重拳
            ['X'],                  # 12: 轻脚
            ['Y'],                  # 13: 中脚
            ['Z'],                  # 14: 重脚
            # 复合动作 (解决缩墙角的杀手锏)
            ['RIGHT', 'C'],         # 15: 前插重拳 (一般距离远)
            ['DOWN', 'Z'],          # 16: 扫堂腿 (下段击倒，争取起身时间逃跑)
            ['A', 'B', 'C'],        # 17: 三拳同按 (某些角色的特殊逃脱技)
        ]
        
        self._actions = []
        for action in actions:
            arr = np.array([0] * 12, dtype=np.int8)
            for button in action:
                arr[buttons.index(button)] = 1
            self._actions.append(arr)
        
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]