## 2021 NIPS
[Ye, Feiyang, et al. "Multi-objective meta learning." Advances in Neural Information Processing Systems 34 (2021): 21338-21351.](https://arxiv.org/abs/2102.07121)

**Notes:**
Source code: https://github.com/Baijiong-Lin/MOML

The article defines the MOBLP (Multi-Objective Bi-Level Optimization Problem) as the following:

$$
\min_{\alpha\in \mathcal{A}, \omega\in\mathbb{R}^p} F(\omega, \alpha) \quad \text{s.t. $\omega \in S(\alpha)$ }
$$

Notice the above $\mathcal{A}$ is not really action space, indeed in the example it givens for reinforcement learning, it is value space for weight.

This paper gives several assumptions, but according to its code, how it calculates the gradients with respect to $\alpha$ (which is the weight in multi task learning) is loading the weights of the model to a temp model named `meta_model`. And do the optimization with validation dataset on `meta_model` (This gradient descent on $\omega$ would connected to the value of $\alpha$). Then do the evaluation again on this, and do the gradient descent on the weights parameter on $\alpha$.

Check this answer for how to calculate the derivative of argmin in this link: https://math.stackexchange.com/questions/2261172/interchange-derivative-and-argmin

The answer in the link mentioned calculating the derivative is ***not*** simply just a exchange. i.e. $g(x, y) = y^2, h(x) = \argmin_{y\in R^+} g(x, y) = \argmin_{y\in R^+} y^2$. Notice this does nothing with variable $x$ thus $\partial_x h=0$. Whereas if you simply exchange $argmin_{y\in R^+}$ and $\partial$, it will become $argmin_{y\in R^+} 0$ which the result is the whole $R^+$. 

Now, considering g with following assumptions:
- minimum exists uniquely *(locally I guess?)* for all $x$, that $g\in C^2$
- $\partial_y^2g(x,y) > 0$ at $y = h(x)$.

Then it is not hard to get by definition $\partial_2 g(x, h(x)) = 0$ where $\partial_2$ is the derivative with respect to the second argument. Therefore we would have $\frac{\partial[\partial_2 g(x, h(x))]}{\partial x} = 0$. Which if you just expand it by definition, you would get:
$$
\partial_1 \partial_2 g(x, h(x)) + h'(x)\partial_2^2 g(x, h(x)) = 0
$$

Then, with our assumption that $\partial_y^2g(x,y) > 0$ at $y = h(x)$. We can simply solve the $h'(x)$.