
from pickletools import bytes1
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import random
from ray import tune
from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=100, default_timesteps=100000
)

parser.add_argument(
    "--default_timesteps",  # The new argument name
    type=int,
    default=1000000,  # Set default if not provided
    help="Total number of timesteps to train the model."
)
parser.add_argument(
    "--default_iters",  # The new argument name
    type=int,
    default=50,  # Set default if not provided
    help="Total number of iters."
)

class PongEnv(gym.Env):
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        
        self.screen_width = 800
        self.screen_height = 600

        self.paddle_height = 100
        self.paddle_width = 20

        self.ball = [400,300]
        self.ball_move = [10, 10]
        self.ball_radius = 10

        self.direction = 0  #0-3
        self.position_agent1 = 300
        self.position_agent2 = 300
        self.hit_back = False
        self.perfect_hit = False
        self.ballEventSource = -1  # which agent's side is the ball on currently

        self.rewards = {'agent1': 0,'agent2': 0},

        self.ball_out_of_bounds = False
        
        self.action_space_agent1 = Discrete(3)
        self.action_space_agent2 = Discrete(3)
        self.observation_space = Dict(
            {
            'paddle_height': Box(0, self.screen_height, shape=(2,), dtype=np.float32),
            'ball_position': Box([0,0], [self.screen_width, self.screen_height], shape=(2,), dtype=np.float32)
            })

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.position_agent1 = 300
        self.position_agent2 = 300
        return np.array([self.position_agent1, self.action_space_agent2, self.ball[0], self.ball[1]], np.float32), {"env_state": "reset"}

    def step(self, action_agent1, action_agent2):
        assert action_agent1 in [0, 1, 2], action_agent1
        assert action_agent2 in [0, 1, 2], action_agent2
        
        if(action_agent1 == 1 and self.position_agent1 > 20):
            self.current_poz -= 40
            #print("Agent1 Moved up")       
        elif(action_agent1 == 2 and self.position_agent1< self.screen_height - 20):
            self.current_poz += 40
            #print("Agent1 Moved down")

        if(action_agent2 == 1 and self.position_agent2 > 20):
            self.current_poz -= 40
            #print("Agent2 Moved up")
        elif(action_agent2 == 2 and self.position_agent2 < self.screen_height - 20):
            self.current_poz += 40
            #print("Agent2 Moved down")

        self.move_ball()

        terminated = self.hit_back == True
        truncated = False

        if(self.ballEventSource == 1):
            self.rewards["agent1"] = self.calcReward()
        elif (self.ballEventSource == 2):
            self.rewards["agent2"] = self.calcReward()
   
        self.perfect_hit = False
        self.hit_back = False
        infos = {}
        
        return (
            np.array([self.position_agent1, self.action_space_agent2, self.ball_x, self.ball_y], np.float32),
            {
            'agent1 reward': self.rewards["agent1"],
            'agent2 reward': self.rewards["agent2"]
            },
            terminated,
            truncated,
            infos,
        )

    #-------------------------------------------------------------------------------gamelogic-------------------------------------------------------------------------------------------------------
    def calcReward(self):
        terminated = self.hit_back == True
        if(self.ballEventSource == 1):
            agentId = 1
        else:
            agentId = 2

        if (terminated):
            if(self.perfect_hit):
               reward = random.uniform(1.3, 2) 
               print(f"Agent {agentId}:Perfect hit!")
            else:
               reward = random.uniform(0.4, 0.9)
               print("Close call.")
        elif (self.ball_out_of_bounds):
            reward = -0.3
            self.ball_out_of_bounds = False
            print("Miss")
        else :
            reward = 0
        return reward

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
        if(abs(self.ball_y - self.current_poz) <= 20):
            self.perfect_hit = True


    def paddle_collision(self): #if
        collided = False
        bx0 = self.ball[0] - self.ball_radius    #ball borders
        bx1 = self.ball[0] + self.ball_radius    
        by0 = self.ball[1] - self.ball_radius
        by1 = self.ball[1] + self.ball_radius
            
        if(self.ballEventSource == 1):
            px0 = 50 - self.paddle_width/2          #paddle borders
            px1 = 50 + self.paddle_width/2
            py0 = self.position_agent1 - self.paddle_height/2
            py1 = self.position_agent1 + self.paddle_height/2
            if( (bx0 < px1 and bx0 > px0) and 
               ( (by0 > py0 and by0 < py1) or (by1 > py0 and by1 < py1) )):
                collided = True
        else:
            px0 = self.screen_width - 50 - self.paddle_width/2          #paddle borders
            px1 = self.screen_width - 50 + self.paddle_width/2
            py0 = self.position_agent2 - self.paddle_height/2
            py1 = self.position_agent2 + self.paddle_height/2
            if( (bx1 > px0 and bx1 < px1) and 
               ( (by0 > py0 and by0 < py1) or (by1 > py0 and by1 < py1) )):
                collided = True
        return collided
   

    def move_ball(self):
        bx0 = self.ball[0] - self.ball_radius    #ball borders
        bx1 = self.ball[0] + self.ball_radius
        by0 = self.ball[1] - self.ball_radius
        by1 = self.ball[1] + self.ball_radius
        x = self.ball_move[0]
        y = self.ball_move[1]

        if(self.ball[0] < self.screen_width/2):
            self.ballEventSource = 1
        elif(self.ball[0] > self.screen_width/2):
            self.ballEventSource = 2
        collided = self.paddle_collision()

        if(collided == False):
            if(bx0 - x < 0 or bx0 + x > self.screen_width):          
                self.ball_out_of_bounds = True
                self.reset_ball()
            elif(bx1 + x > self.screen_width):
                self.collision_right()
            elif(by0 - y < 0):
                self.collision_top()
            elif(by1 + y > self.screen_height):
                self.collision_bottom()
        else:
            self.collision_left()

        if(self.direction == 0):
            y = -y
        elif(self.direction == 2):
            x = -x
        elif(self.direction == 3):
            x= -x
            y= -y
        
        self.ball[0] += x
        self.ball[1] += y

    def reset_ball(self):
        rand = random.randint(0, 1)
        if(rand == 0):              #left side ball spawn
            self.direction  = random.randint(0, 1)
            self.ball[0] = 80
        else:
            self.direction  = random.randint(2, 3)
            self.ball[0] = self.screen_width - 80

        randy = random.uniform(50, 550)
        self.ball[1] = randy

from ray.rllib.callbacks.callbacks import RLlibCallback
class MyCallbacks(RLlibCallback):
    def on_episode_end(self, *, algorithm, metrics_logger, result, **kwargs):
        reward_sum = sum(episode.agent_rewards.values())  # Sum of all rewards
        episode.custom_metrics["episode_reward_sum"] = reward_sum
        print(f"Reward sum: {reward_sum}")

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        print(f"Training result: {result}")
        reward_sum = sum(episode.agent_rewards.values())  # Sum of all rewards
        print(f"Reward sum: {reward_sum}")
    

if __name__ == "__main__":
    args = parser.parse_args()

    # Can also register the env creator function explicitly with:
    register_env("pong-env", lambda config: PongEnv())

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            PongEnv,  # or provide the registered string: "corridor-env"
            #env_config={"corridor_length": args.corridor_length},
        )
    )
    # Customize base config further (like number of workers, GPUs, etc.)
    base_config["num_workers"] = 1  # Adjust based on resources
    base_config["framework"] = "torch"  # Or "tf" depending on your setup
    base_config["num_gpus"] = 0  # Set to 1 if you have a GPU and want to use it
    base_config["callbacks"] = MyCallbacks

    stopping_criteria = {
        "num_env_steps_sampled_lifetime": args.default_timesteps,
        "training_iteration": args.default_iters,
        #"time_total_s": 20
    }


    tune.run(
        "PPO",
        config=base_config,
        stop=stopping_criteria,
        checkpoint_at_end=True  # Ensures a checkpoint is saved at the end of training
    )

    run_rllib_example_script_experiment(base_config, args)