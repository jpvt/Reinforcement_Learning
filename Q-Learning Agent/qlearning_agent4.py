import gym
import numpy as np

env = gym.make('MountainCar-v0')

# Q-Learning Settings
LEARNING_RATE = 0.1 #can tweak later
DISCOUNT = 0.95 #Mesure How much we value future actions. Between 0-1
EPISODES = 500
SHOW_EVERY = 50

# We want this Q-Table to be a size that is manageable so...
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 
# ^ This variable you might decide to tweak later
#Now we need to know how big is our range
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#Creating the Q Table :
q_table = np.random.uniform(low = -2, high = 0, size =(DISCRETE_OS_SIZE + [env.action_space.n]))
#Starts randomly a Q-Table that is Every Combination for each action that is possible it is,
# in this case, 20x20x3

def get_discrete_state(current_state):
    discrete_state = (current_state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))




for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    
    while not done:
        action = np.argmax(q_table[discrete_state]) #takes the optimal action
        nu_state, reward, done, _ = env.step(action) #make a step to that action

        nu_discrete_state = get_discrete_state(nu_state) #get new discrete state of the new state
        
        if episode % SHOW_EVERY == 0:
            #print('render')
            env.render()
        
        if not done:
            max_future_q = np.max(q_table[nu_discrete_state]) 
            # ^ selects the Optimal Q-value instead of the Argmax of the Q-value 
            # for the New Q formula
            current_q = q_table[discrete_state + (action, )]#slicing to get the Q-Value

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #new_q formula ^ see image for more details

            q_table[discrete_state+(action, )] = new_q #update the Q-Table with new Q-Value
        elif nu_state[0] >= env.goal_position:
            print(f'We made it on episode {episode}')
            q_table[discrete_state + (action, )] = 0 #Reward for completing things. Nothing !
            
        
        discrete_state = nu_discrete_state

env.close() 
