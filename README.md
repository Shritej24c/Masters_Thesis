# Modeling Human Balance and Gait  



**Understanding Balance in Humans**


In biomechanics, balance is the ability to maintain the line of gravity (vertical line from the center of mass) of a body within the base of support with minimal postural sway.

Sway is the horizontal movement of the centre of gravity when the person is standing still

Postural sway can be observed in 2 dimensional : Mediolateral(ML) or Coronal or Frontal plane and Anterioposterial(AP) or Saggital plane

![image](https://scalar.usc.edu/works/edkp-3/media/1-300x267.png) 


**Objective**

Modelling postural sway of human body while standing bipedally using simplest human mechanical architecture

**Data**

Healthy study subjects were made to stand on a force sensing platform and their movement of Centre of Pressure (CoP) was measured with time

![image](https://ibb.co/9Gpv7pW) 


Implemented DDPG algorithm to model the healthy adult's postural sway characteristics by assuming human as a inverted pendulum (a simplistic mechanical model) with 1 degree of freedom and producing optimum torque to balance it with respect to the vertical.

![image](https://cecs.anu.edu.au/sites/default/files/resize/u325/2-300x598.jpg)


The whole review of the project can be found in "Thesis_shritej.pdf" and "DDP_Review.pdf".

Environment for an inverted pendulum developed by using OpenAI Gym

Stable Baselines was used for RL agent (DDPG) 

