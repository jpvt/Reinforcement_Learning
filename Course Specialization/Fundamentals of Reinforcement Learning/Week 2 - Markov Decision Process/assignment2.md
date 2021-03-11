# Assignment 2 - Describe three MDPs

For this assignment you will get experience thinking about Markov Decision Processes (MDPs) and how to think about them. You will devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as different from each other as possible.


## 1. Robot Vacuums
---

**Agent**: Robot vacuums

- *States*: sensor readings that the robot takes each time step; cliff sensors, bump sensors, wall sensors, optical encoders, current wheel angle, battery level, dust scanner.

- *Actions*: Go straight, turn wheel right, turn wheel left, and stop.

- *Rewards*: -1 at every time step until the dust scanner tells the capacity is full; -10 if gets stuck in something and it has to be rescued; +10 if the dust scanner is full while battery level is high and +5 if the dust scanner is full while the battery level is low; -5 if the dust scanner is not full and the battery runs out.

## 2. Pong
---

**Agent**: Pong player

- *States*: agent's position, adversary's position, ball's position, agent's score, and adversary's score

- *Actions*: Go up, go down, and stay in position.

- *Rewards*: +1 if it makes a score and doubles it for each consecutive score; -1 if it the adversary makes a score and doubles it of each consecutive score

## 3. Dino Runner
---

**Agent**: T-Rex from the Dinosaur Game

- *States*: agent's score, current pixels in the screen (with distance to the closest objects).

- *Actions*: jump, duck, and do nothing.

- *Rewards*: +5 for each second it stays alive; -1 billion if it hits and object
