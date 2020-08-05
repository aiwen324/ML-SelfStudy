# EM Algo

### Proof of Correctness
Expectation-maximization works to imrpove $Q(\theta|\theta^{(t)})$ rather than directly improving $log_p (X|\theta)$. Here is shown that improvements to the former imply improvements to the latter.

For any $\mathbf{Z}$ with non-zero probability $p(Z|X,\theta)$, we can write 
$$
p(X|\theta) = \frac{p(X,Z|\theta)}{p(Z|X,\theta)}
$$
Then after adding $log$ function on each side, we would have 
$$
log  p(X|\theta)=logp(X,Z|\theta)-logp(Z|X,\theta)
$$
Apply expectation on Z to each side, we would have 
$$
\begin{aligned}
\sum_{\mathbf{Z}} p(Z|X,\theta^{(t)})logp(X|\theta)=&\sum_{\mathbf{Z}} p(Z|X,\theta^{(t)})logp(X,Z|\theta)- \\
&\sum_{\mathbf{Z}} p(Z|X,\theta^{(t)})logp(Z|X,\theta) \\
\end{aligned}
$$
Notice left side is constant, we could simplify it as:

