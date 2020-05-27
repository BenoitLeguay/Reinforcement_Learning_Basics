![1590586508219](/home/benoit/Documents/work/reinforcement_learning/photo/1590586508219.png)



### Tabular Setting

Tabular methods refer to problems in which the state and actions spaces are small enough for 
approximate value functions to be represented as arrays and tables

### Function Approximation

On the contrary, when the state (and/or action) space is too large, the tabular methods are not applicable for efficiency and memory reasons. Indeed, it means that we would have to visit the entire state-action space and learn a value for each of them which would take a very large number of episodes, thus the optimal value function would be intractable. Also that would imply storing so many values in memory. I didn't even mention the case where the state (and/or action) space is continuous. In such a space the number of value to learn is infinite.  

In order to address this shortcoming, we can adopt a new approach based on the features of each state. The aim is to use these set of features to generalise the estimation of the value at states that have similar features.

$F(s) = \hat {V}(s)$

There are many function approximator types:

- Linear combinations of features
- Neural networks
- Decision Tree
- Nearest neighbor