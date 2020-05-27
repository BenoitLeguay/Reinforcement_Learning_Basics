# Policy Based Method

In policy-based methods, instead of learning a value function that  tells us what is the expected sum of rewards given a state and an  action, we learn directly the policy function that maps state to action  (select actions without using a value function).

It means that we directly try to optimize our policy function π without worrying about a  value function. We’ll directly parameterize π (select an action without a  value function).

## Algorithm(s) implemented

### REINFORCE



![REINFORCE algorithm with discounted rewards – where does gamma^t ...](https://i.stack.imgur.com/Acbup.png)



**Ressources:**

- https://link.springer.com/content/pdf/10.1007/BF00992696.pdf 
