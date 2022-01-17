# Modeling Human Balance and Gait  

My Master's project on Balance and Gait in Humans(involving Computational Neuroscience and Deep Reinforcement Learning)

**Objective** - Modelling postural sway of human body while standing bipedally using simplest human mechanical architecture

**Definition** : Sway is the horizontal movement of the centre of gravity when the person is standing still

In biomechanics, balance is the ability to maintain the line of gravity (vertical line from the center of mass) of a body within the base of support with minimal postural sway.

Postural sway can be affected in 2 planes, namely, Saggital and Coronal planes 

![image](https://i0.wp.com/biologydictionary.net/wp-content/uploads/2017/02/Planes-of-Body.jpg) ![image](http://scalar.usc.edu/works/edkp-3/media/1-300x267.png)



Implemented DDPG algorithm to model the healthy adult's postural sway characteristics by assuming human as a inverted pendulum (a simplistic mechanical model) with 1 degree of freedom and producing optimum torque to balance it with respect to the vertical.

![image](https://cecs.anu.edu.au/sites/default/files/resize/u325/2-300x598.jpg)


The whole review of the project can be found in "Thesis_shritej.pdf" and "DDP_Review.pdf".

Environment for an inverted pendulum developed by using OpenAI Gym

Stable Baselines was used for RL agent (DDPG) 

