#from tkinter.tix import BALLOON
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)

class PongEnv(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        
        self.screen_width = 800
        self.screen_height = 600

        self.py0 = 250
        self.py1 =350
        self.px0 = 40
        self.px1 = 60

        self.ball_x = 400
        self.ball_y = 300
        self.ball_move_x = 9
        self.ball_move_y = 9
        self.by0 = 290
        self.by1 = 310
        self.bx0 = 390
        self.bx1 = 410

        self.direction = 0  #0-3
        self.current_poz = 300
        self.hit_back = False

        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([
            20,     #paddle y axis
            0,      #ball x poz
            0       #ball y poz
            ]),
            high=np.array([
            580,    #paddle y axis
            800,    #ball x poz
            600     #ball y poz
            ]),
        )

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.current_poz = 300
        return np.array([self.current_poz], np.float32), {"env_state": "reset"}

    def step(self, action):
        assert action in [0, 1, 2], action
        #move up
        if(action == 1 and self.current_poz > 20):
            self.current_poz -= 40
        #move down
        elif(action == 2 and self.current_poz < 580):
            self.current_poz += 40

        self.move_ball()

        terminated = self.hit_back == True
        truncated = False
        
        reward = random.uniform(0.5, 1.5) if terminated else -0.01
        infos = {}
        return (
            np.array([self.current_poz], np.float32),
            reward,
            terminated,
            truncated,
            infos,
        )
    #-------------------------------------------------------------------------------gamelogic-------------------------------------------------------------------------------------------------------

    def collision_top(self):       
        if(self.direction == 3):
            self.direction = 2
        else: self.direction = 1

    def collision_bottom(self):
        if(self.direction == 1):
            self.direction = 0
        else: self.direction = 3

    def collision_right(self):
        if(self.direction == 0):
            self.direction = 3
        else: self.direction = 2

    def collision_left(self):
        if(self.direction == 3):
            self.direction = 0
        else: self.direction = 1
        self.hit_back = True


    def paddle_collision(self):
        collided = False

        if( (self.bx0 < self.px1 and self.bx0 > self.px0) and 
           ( (self.by0 > self.py0 and self.by0 < self.py1) or (self.by1 > self.py0 and self.by1 < self.py1) )):
            collided = True
            self.collision_left()
        return collided
   

    def move_ball(self):
        x = self.ball_move_x
        y = self.ball_move_y
        collided = self.paddle_collision()
        if(collided == False):
            if(self.bx0 - x < 0):
                self.reset_ball()
            elif(self.bx1 + x > self.screen_width):
                self.collision_right()
            elif(self.by0 - y < 0):
                self.collision_top()
            elif(self.by1 + y > self.screen_height):
                self.collision_bottom()

        if(self.direction == 0):
            y = -y
        elif(self.direction == 2):
            x = -x
        elif(self.direction == 3):
            x= -x
            y= -y

        self.ball_x += x
        self.ball_y += y

    def reset_ball(self):
        rand = random.randint(0, 1)
        self.direction = rand
        self.ball_x = 400
        self.ball_y = 300
        
