# Actor-Critic Method

Actor-critic methods are TD methods that have a separate memory structure to explicitly represent the policy independent of the value function.  The policy structure is known as the *actor*, because it is used to select actions, and the estimated value function is known as the *critic*, because it criticizes the actions made by the actor.  Learning is always on-policy: the critic must learn about and critique whatever policy is currently being followed by the actor.  The critique takes the form of a TD error.  This scalar signal is the sole output of the critic and drives all learning in both actor and critic.

1. The “Critic” estimates the value function. This could be the action-value (the Q value) or state-value (the V value).

2. The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with [policy gradients](https://github.com/BenoitLeguay/Reinforcement_Learning/tree/master/policy_based_methods)).


### Algorithm(s) implemented

##### A2C

![RL introduction: simple actor-critic for continuous actions](https://miro.medium.com/max/1676/1*Mymsb9uzxOvPJf_oA1jjBQ.jpeg)

*Ressource:*

- https://arxiv.org/pdf/1602.01783.pdf

##### QAC

![Understanding Actor Critic Methods and A2C - Towards Data Science](https://miro.medium.com/max/5734/1*BVh9xq3VYEsgz6eNB3F6cA.png)

*Ressource:*

- https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf

##### PPO

![Proximal Policy Optimization — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)

*Ressource:*

- https://arxiv.org/pdf/1707.06347.pdf

##### A3C

*Ressource:*

- https://arxiv.org/pdf/1602.01783.pdf