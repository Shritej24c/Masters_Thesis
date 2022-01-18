# Modeling Human Balance and Gait  



**Understanding Balance in Humans**


In biomechanics, balance is the ability to maintain the line of gravity (vertical line from the center of mass) of a body within the base of support with minimal postural sway.

Sway is the horizontal movement of the centre of gravity when the person is standing still

Postural sway can be observed in 2 dimensional : Mediolateral(ML) or Coronal or Frontal plane and Anterioposterial(AP) or Saggital plane

![image](https://scalar.usc.edu/works/edkp-3/media/1-300x267.png) 


**Objective**

Modelling postural sway of human body while standing bipedally using simplest human mechanical architecture

**Data**

Healthy study subjects were made to stand on a force sensing platform and their movement of Centre of Pressure (CoP) was measured with time. View the following image for illustration purposes

 ![image](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/ijerph-18-02696-g001-550.jpeg)
 
 
 Below is the real data of Sway of 4 Healthy Adults and their respective frequency spectrum 
 
 ![alt text](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/Real%20Data.png)
 
 **Modeling**
 
For a rudimentary approach, we assumed a human body as an inverted pendulum where the length of the pendulum is the distance from average human's center of mass to point of contact with the ground.

Due to the similar reward-based learning observed in human movements, we adopted Reinforcement Learning methodology to model the postural sway.

**Reinforcement Learning**

Reinforcement Learning (RL) is the science of decision making. It is about learning the optimal behavior in an environment to obtain maximum reward. This optimal behavior is learned through interactions with the environment and observations of how it responds, similar to video game player playing Mario, exploring the world around him and learning the actions that help them achieve the maximum reward towards the end of the level or turn.
 

![image](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/Mario.png)


**Approach**

States of the RL agent - Angular Displacement with respect to the vertical and Angular Velocity of the pendulum

Actions - Torque applied to the pendulum

Reward - Function of Angular Velocity and Angular Displacment ( R = -(T^2 + 0.1*V*^2) )

![image](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/Block%20Diagram.png)

OpenAI Gym was use to build the RL environment and Stable Baseline to train the Rl agent.

A deep reinforcement learning algorithm called Deep Deterministic Policy Gradient (DDPG) algorithm was used for continuous action space  considering the problem statement


**Results**


Below you can see how sway and torque applied varies with the time. The pendulum is trained for 1000 epochs that is 10 secs

![image](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/Sway%20Result.png)
![image](https://github.com/Shritej24c/Masters_Thesis/blob/master/Images/Torque%20Result.png)








The whole review of the project can be found in "Thesis_shritej.pdf" and "DDP_Review.pdf".

Environment for an inverted pendulum developed by using OpenAI Gym

Stable Baselines was used for RL agent (DDPG) 

