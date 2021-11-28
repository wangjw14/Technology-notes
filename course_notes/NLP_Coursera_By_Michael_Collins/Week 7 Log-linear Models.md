# Week 7 Log-linear Models



### Log-linear models

- Log-linear models allow us to use many other "features" of context, for example, the author, the part of speech of previous word, the suffix of the previous word, etc,.

- Definition

  - input domin $\mathcal{X}$ 

  - finite label set $\mathcal{Y}$ 

  - aim to provide a conditional probability $p(y|x)$ for any $x,y$ where $x\in \mathcal{X},y\in\mathcal{Y}$ 

  - A feature is a function $f_k(x,y)\in \Bbb{R}$ (Often binary features or indicator functions $f_k(x,y)\in\{0,1\}$)

  - Then we have $m$ features $f_k$ for $k=1...m$ and a feature vector $f(x,y)\in\Bbb R^m$ for any $x,y$

  - Given features $f_k(x,y)$ for $k=1...m$ , also define a parameter vector $v\in \Bbb R^m$ and each $(x,y)$ pair is then mapped to a "score"
    $$
    v\cdot f(x,y) = \sum_kv_kf_k(x,y)
    $$

  - Define 
    $$
    p(y|x;v)=\frac{e^{v\cdot f(x,y)}}{\sum_{y^\prime\in\mathcal {Y}}e^{v\cdot f(x,y^\prime)}}
    $$
    

- For a language model, we would probably introduce one trigram feature for every trigram seen in the training data (Do NOT includes trigrams not seen in training data). For all trigram $(u,v,w)$ seen in training data, create a feature

$$
f_{N(u,v,w)}(x,y)=
\begin{cases}
1 & \text{if } y=w,w_{i-2}=u,w_{i-1}=v \\
0 & \text{otherwise}
\end{cases}
$$

​		where $N(u,v,w)$ is a function that maps each $(u,v,w)$ trigram to a different integer.

- Why the name?
  $$
  \log p(y|x;v)= v\cdot f(x,y) - \log\sum_{y^\prime\in\mathcal Y}e^{v\cdot f(x,y^\prime) }
  $$
  the first term is the linear term, and the second term is a function of $x$ and doesn't depend on $y$. Hence we call the model log-linear model.

### Parameter estimation in log-linear models

- Maximun-likelihood estimates given training sample $(x^{(i)},y^{(i)})$ for $i=1...n$, each $(x^{(i)},y^{(i)})\in \mathcal X \times \mathcal Y $:
  $$
  v_{ML}=\arg\max_{v\in\Bbb R^m} L(v)
  $$
  where


$$
\begin{align}
L(v)&= \sum_{i=1}^n\log p(y^{(i)}|x^{(i)};v) \\ 
&= \sum^n_{i=1}v\cdot f(x^{(i)},y{(i)})- \sum^n_{i=1}\log\sum_{y^\prime\in\mathcal{Y}}e^{v\cdot f(x^{(i)},y^\prime)}
\end{align}
$$

- Gradient Ascent Methods

  Need to maximize $L(v)$ where
  $$
  \frac{dL(v)}{dv_k}=\sum^n_{i=1}f_k(x^{(i)},y^{(i)})-\sum_{i=1}^n\sum_{y^\prime\in\mathcal{Y}}f_k(x^{(i)},y^\prime)p(y^\prime|x^{(i)};v)
  $$
  **Initialization:** $v=0$

  **Iterate until convergence:**

  - Calculate $\Delta=\frac{dL(v)}{dv}$
  - Calculate $\beta_*=\arg\max_\beta L(v+\beta\Delta)$ (Line search)
  - Set $v \leftarrow v+\beta_*\Delta$

- Calculate gradient using conjugate gradient methods (such as LBFGS)


### Smoothing/regularization in log-linear models

- Regularization

  - Modified loss function
    $$
    L(v)=\sum^n_{i=1}v\cdot f(x^{(i)},y{(i)})- \sum^n_{i=1}\log\sum_{y^\prime\in\mathcal{Y}}e^{v\cdot f(x^{(i)},y^\prime)}-\frac\lambda 2\sum_{k=1}^mv_k^2
    $$

  - Calculating gradients
    $$
    \frac{dL(v)}{dv_k}=\sum^n_{i=1}f_k(x^{(i)},y^{(i)})-\sum_{i=1}^n\sum_{y^\prime\in\mathcal{Y}}f_k(x^{(i)},y^\prime)p(y^\prime|x^{(i)};v) -\lambda v_k
    $$

- Experiments with Gaussian priors

  - In regular (unregularized) log-linear models, if all n-gram features are included, then it's equivalent to maximum-likelihood estimates
    $$
    q(w_i|w_{i-2},w_{i-1})=\frac{Count(w_{i-1},w_{i-1},w_i)}{Count(w_{i-2},w_{i-1})}
    $$

  - With regularization, get very good results. Performd as well as or better than standardly used "discounting methods"

  - Downside: computing $\sum_we^{f(w_{i-2},w_{i-1},w_i)\cdot v}$ is **SLOW**.

