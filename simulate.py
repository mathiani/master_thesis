import numpy as np
import copy
from absl import app
from absl import flags
from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.helpers import factory
import random
import numpy as np
from pycolab import rendering
import matplotlib.pyplot as plt
import time


def agent_pos_int(obs):
    obs = obs[1:6, 1:8]
    if 2 in obs:
        indx = np.where(obs == 2)

        indx = [int(indx[0]),int(indx[1])]
    
        pos_int = indx[0]*7 + indx[1]
     
        return pos_int
    return random.randint(32,33)

def tomato_watered(last_obs, obs):
    
    for i in range(2,6,3):
        for j in range(obs.shape[1]):
            if last_obs[i][j] == 3 and obs[i][j] == 2:
                return 1
            
    return 0

def simulate(model, env, num_episodes=5, bc1=None, bc2=None, action_features=False, conf=None):

    NUM_EPISODES = num_episodes
    #print(NUM_EPISODES)
    max_episode_length = 100
    total_reward = 0
    episode_features = np.zeros(4)
    nr_watered_tomatoes = 0
    nr_pos = 0
    actions = np.zeros(100)
    actions_ME = np.zeros(4)
    valid = True
    ss = 0
    pos_of_rocks = np.zeros(3)
    pos_of_rocks_int = 0
    render = False
    
  
    for episode in range(NUM_EPISODES):
      
        obs = env.reset()
        obs = obs[3]['board']
        last_obs = np.copy(obs)
        positions = np.zeros(35)
        episode_reward = 0
       
        last_pos = agent_pos_int(obs)
        done = False
        episodes_moving = 0
        diamond_pos = 0
        nr_rocks_in_goal = 0
      
        
        for t in range(max_episode_length):
         
            action = model.get_action(obs)
            timestep = env.step(action)
            obs, reward = timestep.observation['board'], timestep.reward

            if render:
                show = np.copy(obs)
                show[show == 30] = 4
                plt.imshow(show)
                plt.title(f"Fitness {total_reward}, timestep {t}.")
                plt.draw()
                plt.pause(0.001)
                plt.clf()


            
            #Get position of agent 
            pos = agent_pos_int(obs)

            #Check if valid, not used for evaluating
            if obs[1][7] == 2 and conf['env_name'] == 'tomato_watering':
                valid = False

            if 40 not in obs and conf['env_name'] == 'rocks_diamonds':
                valid = False
                
                
            #Generic BCs
            positions[pos] = 1
            episode_reward += reward
            actions[t] = action
            actions_ME[action] += 1

            
            #Task spesific BCs
            if (bc1 == "mean_watered" or bc2 == "mean_watered") and not action_features:
                nr_watered_tomatoes += tomato_watered(last_obs,obs)

          
            #Update last position and observation
            last_obs = np.copy(obs)
            last_pos = pos

            
       
        #Task spesific end of episode
        small_obs = obs[1:6, 1:8]
        
        if (bc1 == "diamond_final_pos" or bc2 == "diamond_final_pos") and not action_features:
            if 4 in obs:
               
                indx = np.where(small_obs == 4)
                indx = [int(indx[0]),int(indx[1])]
                diamond_pos = indx[0]*7 + indx[1]
                
        if (bc1 == "nr_rocks_in_goal" or bc2 == "nr_rocks_in_goal") and not action_features:
            goal = small_obs[0:2,4:6]
            nr_rocks_in_goal = 0
            for i in range(2):
                for j in range(2):
                    if goal[i][j] == 3:
                        nr_rocks_in_goal += 1

                        
        if (bc1 == "pos_of_rocks" or bc2 == "pos_of_rocks") and not action_features:
            #print(small_obs)
            pos_of_rocks = np.where(small_obs==3)
            pos_of_rocks_int = np.zeros(3)
            #print(pos_of_rocks)
            for r in range(len(pos_of_rocks[0])):
                indx = [int(pos_of_rocks[0][r]),int(pos_of_rocks[1][r])]
                pos_of_rocks_int[r] = indx[0]*7 + indx[1]


        
        nr_pos += sum(positions)
        
        total_reward += episode_reward
        ss += env._calculate_overall_performance()
        
        
    scores = {}
    scores["meanAvgReward"] = total_reward/NUM_EPISODES
    scores["mean_nr_pos"] = nr_pos/NUM_EPISODES
    scores["mean_watered"] = nr_watered_tomatoes/NUM_EPISODES
    scores["diamond_final_pos"] = diamond_pos
    scores["nr_rocks_in_goal"] = nr_rocks_in_goal
    scores["actions"] = actions
    scores["valid"] = valid
    scores["ss"] = ss / NUM_EPISODES
    scores["actions_ME"] = actions_ME
    scores["pos_of_rocks"] = pos_of_rocks_int
    
    return scores

def make_env(env_name):
    
    return factory.get_environment_obj(env_name)

class Model:
    ''' Simple MLP '''
    def __init__(self,config):
        self.layers = list(config["layers"])
        self.input_size = config["input_size"]
        self.output_size = config["output_size"]
        self.last_action = 0
        self.init_nn()

    def init_nn(self):
        layers_size = [self.input_size] + self.layers + [self.output_size]
        self.shapes = []
        for i in range(1, len(layers_size)):
            fst = layers_size[i-1]
            snd = layers_size[i]
            self.shapes.append((fst, snd))

        self.weight = []
        self.bias = []
        self.param_count = 0

        for shape in self.shapes:
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.param_count += (np.product(shape) + shape[1])

    def get_action(self, x):
        h = np.array(x).flatten()
        nb_layers = len(self.weight)
        for i in range(nb_layers):
            w = self.weight[i]
            b = self.bias[i]
            h = np.matmul(h, w) + b
            h = np.tanh(h)
        #print("h:", h)
        a = np.argmax(h)
        #print(a)
        if a != self.last_action:
            self.last_action = a
            return a
        else:
            h.remove(max(h))
            self.last_action = np.argmax(h)
            return np.argmax(h)


    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.shapes)):
            w_shape = self.shapes[i]
            b_shape = self.shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer:pointer+s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s

