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

The answer in the link mentioned calculating the derivative is ***not*** simply just a exchange. i.e. $g(x, y) = y^2, h(x) = \arg\min_{y\in R^+} g(x, y) = \arg\min_{y\in R^+} y^2$. Notice this does nothing with variable $x$ thus $\partial_x h=0$. Whereas if you simply exchange $\arg\min_{y\in R^+}$ and $\partial$, it will become $\arg\min_{y\in R^+} 0$ which the result is the whole $R^+$. 

Now, considering g with following assumptions:
- minimum exists uniquely *(locally I guess?)* for all $x$, that $g\in C^2$
- $\partial_y^2g(x,y) > 0$ at $y = h(x)$.

Then it is not hard to get by definition $\partial_2 g(x, h(x)) = 0$ where $\partial_2$ is the derivative with respect to the second argument. Therefore we would have $\frac{\partial[\partial_2 g(x, h(x))]}{\partial x} = 0$. Which if you just expand it by definition, you would get:

$$
\partial_1 \partial_2 g(x, h(x)) + h'(x)\partial_2^2 g(x, h(x)) = 0
$$

Then, with our assumption that $\partial_y^2g(x,y) > 0$ at $y = h(x)$. We can simply solve the $h'(x)$.

## Classics
[Baydin, Atilim Gunes, et al. "Automatic differentiation in machine learning: a survey." Journal of Marchine Learning Research 18 (2018): 1-43.](https://arxiv.org/abs/1502.05767v4)

**NOTES:**

**IMPORTANT: AD IS NOT SYMBOLIC DIFFERENTIATION**

Consider the following example, we define

$$
l_{n+1} = 4l_n(1-l_n),\; l_1 = x
$$

Consider $\frac{dl_n}{dx}$. It will be much easy to calculate the derivative recursively. i.e. $\frac{dl_{n+1}}{dx} = \frac{dl_{n+1}}{dl_n}\frac{dl_n}{dl_{n-1}}\cdots\frac{dl_1}{dx}$.

This paper mentioned a representation of computation called ***evaluation trace*** of elementary operaions which forms the basis of the AD techniques. By using this, AD can be applied to regular code with minimal change, allowing branching, loops, and recursion.

AD has two mode: **Forward Mode** and **Reverse Mode**.

**Forward Mode**:

Forward mode is really just trying to calculate the partial derivative for one variable at once. Consider the function $f: R^n -> R^m$. It can be treated as calculate a column of Jacobian matrix $\frac{\partial y}{\partial x_i}$ (which is a $mx1$ vector)

Consider this example:

$$
\begin{align*}
v_{-1} &= x_1 \\
v_0 &= x_2 \\
v_1 &= \ln v_{-1} \\
v_2 &= v_{-1} \times v_0 \\
v_3 &= \sin v_0 \\
v_4 &= v_1 + v_2 \\
v_5 &= v_4 - v_3
\end{align*}
$$

Basically it is calculating $\dot{v}_i = \frac{\partial v_i}{\partial x_i}$. For example: $\dot{v}_2 = v_0\dot{v}_{-1} + v_{-1}\dot{v}_0$

For forward mode, they introdcue another interesting idea called *dual number*. **(Actually you could ignore this extra definition, it's not useful, it's just trying to map the algorithm implemented in the program to a math formula)**. Consider the expression $a + b\epsilon$ where $a, b\in R$, and $\epsilon$ is a symol such that $\epsilon^2 = 0, \epsilon\neq 0$. We call $\epsilon$ dual number. We can represent (map) this in a matrix space, consider the following:

$$
\begin{align*}
\begin{bmatrix}
a & b \\
0 & a
\end{bmatrix}
\begin{bmatrix}
a & b \\
0 & a
\end{bmatrix}
= 
\begin{bmatrix}
a^2 & ab + ba \\
0 & a^2
\end{bmatrix}
\end{align*}
$$

From what we want $(a + b\epsilon)(a + b\epsilon) = a^2 + (ab + ba)\epsilon + b^2\epsilon^2 = a^2 + (ab + ba)\epsilon$ by $\epsilon^2 = 0$. In this case we can view $a + b\epsilon$ as $a\cdot 1 + b\cdot \epsilon$. Then 

$$
\begin{align*}
1 = 
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}, 
\epsilon = 
\begin{bmatrix}
0 & 1 \\
0 & 0   
\end{bmatrix}
\end{align*}
$$

Indeed, there are other representation of $\epsilon$.

$$
\begin{align*}
\epsilon = 
\begin{bmatrix}
    a & b \\
    c & -a
\end{bmatrix}
\end{align*}
$$

where $a^2 + nv = 0$ except when $a=b=c=0$ (because this case $\epsilon = 0$ would conflict with our assumption).

Now consider the differentiation case. First let's consider a polynomial, with $P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_nx^n$. Then we have

$$
\begin{align*}
P(a + b\epsilon) 
&= p_0 + p_1(a + b\epsilon) + \cdots + p_n(a + \epsilon)^n \\
&= p_0 + p_1a + p_2a^2 + \cdots + p_na^n + p_1b\epsilon + 2p_2ab\epsilon + \cdots + np_na^{n-1}b\epsilon \\
&= P(a) + bP'(a)\epsilon
\end{align*}
$$

Now, this means we can extend any **analytic** real function to the dual numbers by looking at its Taylor series. Recall:

$$
\begin{align*}
f(a + x) = f(a) + f'(a)x + \frac{f^{(2)}(a)x^2}{2!} + \cdots + \frac{f^{(n)}(a)x^n}{n!} + \cdots
\end{align*}
$$

Then by letting $x = b\epsilon$, we get:

$$
\begin{align*}
f(a + b\epsilon) 
&= \sum_{n=0}^{\infty}\frac{f^{(n)}(a)b^n\epsilon^n}{n!}
&= f(a) + f'(a)b\epsilon
\end{align*}
$$

In another sense, if we have $a + b\epsilon$ as $v + \dot{v}\epsilon$. Then we have $f(v + \dot{v}\epsilon) = f(v) + f'(v)\dot{v}\epsilon$. Therefore, we could use dual numbers as data strucutres for carrying the tangent value together with the primal. (Maybe just like a tuple same as complex nubmer). The chain rule should work as expected as you can check.

$$
\begin{align*}
f(g(v + \dot{v}\epsilon))
&= f(g(v) + g'(v)\dot{v}\epsilon) \\
&= f(g(v)) + f'(g(v))g'(v)\dot{v}\epsilon
\end{align*}
$$

The coefficient of $\epsilon$ on the right-hand side is exactly the derivative of the composition of $f$ and $g$.

**Reverse Mode:**

Reverse mode has two phases, first we calculate the value as forward pass, then we calculate the gradient by backward trace. 

The paper gives an example of $y = f(x_1, x_2) = ln(x_1) + x_1x_2 - sin(x_2)$. The main point here is carry over. It is not surprising to find out the gradient of a variable can be stacked due to the chain rule. Consider the following *(Check the right arrow later after you read the next two paragraphs)*:

$$
\begin{align*}
v_{-1} &= x_1 \\
v_0 &= x_2 \\
v_1 &= \ln v_{-1} \\
v_2 &= v_{-1} \times v_0 \qquad \Rightarrow \bar{v}_0 = \bar{v}_0 + \bar{v}_2v_{-1}\\
v_3 &= \sin v_0 \qquad \Rightarrow \bar{v}_0 = \bar{v}_0 + \bar{v}_3\cos v_0\\
v_4 &= v_1 + v_2 \\
v_5 &= v_4 - v_3
\end{align*}
$$

Now consider $\bar{v}_i = \frac{\partial y}{\partial v_i}$, let's take an example of $\frac{\partial y}{\partial v_0}$. By chain rule: we know $\frac{\partial y}{\partial v_0} = \frac{\partial y}{\partial v_2} \frac{\partial v_2}{\partial v_0} + \frac{\partial y}{\partial v_3}\frac{\partial v_3}{\partial v_0}$

Now if we do backward propagation, first we can initialize $\bar{v}_0 = 0$. The calculation of $\bar{v}_0$ can be stacked up as following (From top to bottom, since we will calculate $\bar{v}_n$ first):

$$
\bar{v}_0 = \bar{v}_0 + \bar{v}_3\frac{\partial v_3}{\partial v_0} \qquad\text{and}\qquad \bar{v}_0 = \bar{v}_0 + \bar{v}_2\frac{\partial y}{\partial v_0}
$$

Therefore, we can see that one application of the reverse mode is sufficent to compute the full graident of a scalar function: $\nabla f = \left(\frac{\partial y}{\partial x_1} \frac{\partial y}{\partial x_2}\cdots \frac{\partial y}{\partial x_n}\right)$

**Reference:**
- https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
- https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
- https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/

