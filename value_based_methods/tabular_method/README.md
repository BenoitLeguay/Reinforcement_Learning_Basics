# Value Based Method in a tabular setting

In value-based methods, we estimate how good to be in a state. We take actions for the next state that will collect the highest total rewards. Then we infer the optimal policy by taking the path that maximize the sum of (discounted in the continuous case) rewards with: 

- $V^*(s) = \max_{\pi}(V^\pi(s)) \> \forall \> s \> in \>S$
- $V^*(s) = \max_{a}(Q(s, a)) \> \forall \> s \> in \>S$

As you can see, we can use both Q-value function and Value function. 

- $V^\pi(s)$ is the state-value function of MDP (Markov Decision Process). It's the expected return starting from state $s$ following policy $\pi$. Mathematically it is known that way: 

   $V^\pi(s) = E_\pi[G_t | S_t=s]$


- $Q^\pi(s, a)$ is the action-value function. It is the expected return starting from state $s$, following policy $\pi$, taking action $a$. It's focusing on the particular action at the particular state. Mathematically it is known that way:

  $Q^\pi(s, a) = E_\pi[G_t | S_t=s, A_t=a]$	


It exists a relationship between these functions when following policy $\pi$: 

$V^\pi(s) = \sum_{a \in A} \pi(a, s) * Q^\pi(s, a)$


###  Algorithm(s) implemented

##### Sarsa

![Introduction to Reinforcement Learning (Coding SARSA) — Part 4](https://miro.medium.com/max/1952/1*7WZZgbJQr5lh86LRB2pbVg.png)   



*Ressource*: 

- http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf (original paper)
- http://incompleteideas.net/book/first/ebook/node64.html (from *Reinforcement Learning: An Introduction* by R. Sutton and A. Barto)

##### Q-Learning

![Lei Mao's Log Book – On-Policy VS Off-Policy in Reinforcement Learning](https://leimao.github.io/images/blog/2019-03-14-RL-On-Policy-VS-Off-Policy/q-learning.png)



*Ressource*: 

- https://link.springer.com/content/pdf/10.1007/BF00992698.pdf (original paper)

- http://incompleteideas.net/book/first/ebook/node65.html (from *Reinforcement Learning: An Introduction* by R. Sutton and A. Barto)



##### Expected Sarsa

![Temporal-Difference Methods - 知乎](https://pic1.zhimg.com/80/v2-729424eb422e2972f8258d4bb6606354_1440w.jpg)



*Ressource*: 

- http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf

##### Sarsa($\lambda$)

![Implementing SARSA(λ) in Python](https://naifmehanna.com/assets/img/SARSALAMBDA2.png)



*Ressource*: 

- http://incompleteideas.net/book/first/ebook/node75.html (from *Reinforcement Learning: An Introduction* by R. Sutton and A. Barto)