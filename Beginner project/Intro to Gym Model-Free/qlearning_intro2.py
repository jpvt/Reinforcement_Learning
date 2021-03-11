import gym #Inclui pacote Gym, essencial para RL

env = gym.make('MountainCar-v0') #Initialize environment
env.reset() #Resets Env

print(env.observation_space.high) #Highest values for all the observations
print(env.observation_space.low) #Lowest values for all the observations
print(env.action_space.n) #How many actions we can take

# We want this Q-Table to be a size that is manageable so...
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # This variable you might decide to tweak later

done = False #sets finished state as False
random_act = 2 #For this problem there is 3 actions, for example we initially choose 2

while not done:
    action = random_act # define action
    nu_state, reward, done, _ = env.step(action) #New state for the next action, the reward of being in that state and the update of the done object
    print(nu_state)#We want to "bucket" this information into discrete values
    env.render() #Render the graphic env

env.close() #close env
