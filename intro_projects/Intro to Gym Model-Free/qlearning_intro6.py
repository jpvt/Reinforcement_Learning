import gym #Inclui pacote Gym, essencial para RL
import numpy as np

env = gym.make('MountainCar-v0') #Initialize environment
env.reset() #Resets Env

print(env.observation_space.high) #Highest values for all the observations
print(env.observation_space.low) #Lowest values for all the observations
print(env.action_space.n) #How many actions we can take

# We want this Q-Table to be a size that is manageable so...
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 
# ^ This variable you might decide to tweak later
#Now we need to know how big is our range
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

print(discrete_os_win_size)

#Creating the Q Table :

q_table = np.random.uniform(low = -2, high = 0, size =(DISCRETE_OS_SIZE + [env.action_space.n]))
#Starts randomly a Q-Table that is Every Combination for each action that is possible it is,
# in this case, 20x20x3

print(q_table.shape)
print(q_table)

done = False #sets finished state as False
random_act = 2 #For this problem there is 3 actions, for example we initially choose 2

while not done:
    action = random_act # define action
    nu_state, reward, done, _ = env.step(action) #New state for the next action, 
                                                 #the reward of being in that state and the update of the done object
    #print(reward, nu_state) #Rewards -1.0 until it makes a right move
    env.render() #Render the graphic env

env.close() #close env
