import gym
import numpy as np
import matplotlib.pyplot

env = gym.make('MountainCar-v0')

# Q-Learning Settings
LEARNING_RATE = 0.1
DISCOUNT = 0.95 
EPISODES = 2000
SHOW_EVERY = 500

# Exploration Settings
epsilon = 0.4
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) 

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#Creating the Q Table :
q_table = np.random.uniform(low = -2, high = 0, size =(DISCRETE_OS_SIZE + [env.action_space.n]))


ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min':[], 'max' : []}

def get_discrete_state(current_state):
    discrete_state = (current_state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))




for episode in range(EPISODES):
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) 
        else:
            action = np.random.randint(0, env.action_space.n)
        nu_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        
        nu_discrete_state = get_discrete_state(nu_state)
        
        if episode % SHOW_EVERY == 0:
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
    
    # Decaying is being done every episode if episode is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(np.min(-SHOW_EVERY:))
        aggr_ep_rewards['max'].append(np.max(-SHOW_EVERY:))

        print(f'Episode: {episode:>5d} Average: {average_reward:>4.lf} 
                Minimum Reward: {np.min(-SHOW_EVERY:)} Maximum Reward: {np.max(-SHOW_EVERY:)}')

env.close() 