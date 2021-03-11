import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

# Q-Learning Settings
LEARNING_RATE = 0.1 #can tweak later
DISCOUNT = 0.95 #Mesure How much we value future actions. Between 0-1
EPISODES = 25000


# We want this Q-Table to be a size that is manageable so...
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 
# ^ This variable you might decide to tweak later
#Now we need to know how big is our range
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#Creating the Q Table :
q_table = np.random.uniform(low = -2, high = 0, size =(DISCRETE_OS_SIZE + [env.action_space.n]))
#Starts randomly a Q-Table that is Every Combination for each action that is possible it is,
# in this case, 20x20x3

'''
def get_discrete_state(current_state):
    discrete_state = (current_state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))
'''
done = False #sets finished state as False
random_act = 2 #For this problem there is 3 actions, for example we initially choose 2

while not done:
    action = random_act
    nu_state, reward, done, _ = env.step(action) 
    #print(reward, nu_state) 
    env.render() 

env.close() 
