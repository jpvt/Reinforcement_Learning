import gym #Inclui pacote Gym, essencial para RL

env = gym.make('MountainCar-v0') #Inicializa ambiente
env.reset() #Resets Env

done = False #sets finished state as False
random_act = 2 #For this problem there is 3 actions, for example we initially choose 2

while not done:
    action = random_act # define action
    nu_state, reward, done, _ = env.step(action) #New state for the next action, the reward of being in that state and the update of the done object
    env.render() #Render the graphic env

env.close() #close env
