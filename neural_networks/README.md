## Overview

Transform the parameters and compare the performance of tasks.

#### task1.py (Vectorized linear regression)


#### task2.py (task1.py + activation function)


#### task3.py (task2.py + neural networks)


### Result

|                  | Task 1     | Task 2    | Task 3     |
|------------------|------------|-----------|------------|
| Accuracy (Train) | 82%        | 83.1%     | 98.3%      |
| Accuracy (Test)  | 87%        | 87%       | 99.0%      |
| Train time(s)    | 0.32488179 | 0.1280529 | 0.23501706 |
| Test time(s)     | 0.00013399 | 0.0001001 | 0.00010704 |

### Summary

  This project has several helpful concepts, such as back-propagation, activation functions, and layered network structure. Unlike task1, which denotes basic linear regression model, task2 and task3 need somewhat sophisticated methods. So, they perform better than the simple one with higher accuracy. 
Without using the built-in functions(Tensorflow etc), I had to type the code manually. 

By doing these, I could understand its operation mechanism more deeply.
I set learning rate as 0.4, somewhat higher than general situation. Because the increasing inclination is generated with this value. If I suppose the alpha as normal(alpha<=0.01), then the second and the third show lower performance than the first one.
