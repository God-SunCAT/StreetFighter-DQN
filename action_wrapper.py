import numpy as np
import gymnasium as gym

class SF2Discrete15(gym.ActionWrapper):
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
            ['A'],                  # 9: 不是拳就是腿, 反正都是攻击
            ['B'],                  # 10: 
            ['C'],                  # 11: 
            ['X'],                  # 12: 
            ['Y'],                  # 13: 
            ['Z'],                  # 14: 
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