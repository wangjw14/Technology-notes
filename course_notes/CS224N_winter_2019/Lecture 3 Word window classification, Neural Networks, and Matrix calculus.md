# Lecture 3: Word window classification, Neural Networks, and Matrix calculus

### Classification review

- Data: $\{x_i,y_i\}$

- Softmax classifer
  $$
  p(y|x)=\frac{\exp(W_y\cdot x)}{\sum_{c=1}^C\exp(W_c\cdot x)}
  $$
  where $W\in\Bbb R^{c\times d}$ is parameter of model. And we want to maximize the probability of correct class $y$, or we can minimize the negative log probability of that class
  $$
  -\log p(y|x)= -\log(\frac{\exp(f_y)}{\sum_{c=1}^C\exp(f_c)})
  $$

- Cross entropy loss

  Instead of minimize the probability of above function, we can minimize the cross entropy loss of model
  $$
  H(p,q)=-\sum_{c=1}^Cp(c)\log q(c)
  $$
  where $p$ is true probability distribution and $q$ is our computed model probability.

  Cross entropy loss over full dataset is
  $$
  J(\theta)=\frac1N\sum_{i=1}^N-\log(\frac{\exp(f_{y_i})}{\sum_{i=1}^C\exp(f_{c})})
  $$
  (Only retain the items that are not equal to 0)

### Neural networks introduction

- Softmax ($\approx$ logistic regression) alone not very powerful. Softmax only gives linear decision boundaries and this can be quite limiting. Neural networks can learn much more complex functions annd nonlinear decision boundaries.

- Classification in NLP learns **both** conventional parameters and word representations.
  $$
  \theta\in\Bbb R^{Cd+Vd}
  $$

- Understanding neural networks

  - A neuron can be a binary logistic regression unit
  - A neural network runs several logistic regressions at the same time
  - Without non-linearities, deep neural networks can't do anything more than a linear transform

### Named Entity Recognition

- NER
  - The task: **find** and **classify** names in text
  - possible purpose：
    - Tracking mentions of particular entities in documents
    - For question answering, answers are usually named entities
    - A lot of wanted information is really associations between named entities
    - The same techniques can be extended to other slot-filling classifications
  - Often followed by Named Entity Linking/ Canonicalization(规范化) into Knowledge Base.

- Why might NER be hard?
  - Hard to work out boundaries of entities 
  - Hard to know if something is an entities
  - Hard to know class of unknown/novel entity
  - Entity class is ambiguous and depends on context

### Binary true vs. corrupted word window classification

- In general, classifying single words is rarely done. It is more common to classify a word in its context.

- Simplest window classification

  - Idea: classify a word in its context window of neighboring words

  - Train a softmax classifier to classify a center word by taking concatenation of word vectors surrounding it in a window

  - the neural networks for NER (Collobert & Weston (2008, 2011))
    $$
    h =f(Wx+b)\\s=u^Th
    $$
    where $x$ is the concatenation of word vectors and $s$ is an unnormalized score indicating whether the center word is a NER word.

  - **Max-margin loss**

    - Loss function for a single window (minimize)
      $$
      J=\max(0,1-s+s_c)
      $$
      where $s$ is the score for a true window and $s_c$ is the score for a corrupt window

    - Each window with a NER word at its center should have a score +1 higher than any window without a NER word at its center. Trying to minimize $J$ equals trying to get $s\ge s_c+1$

    - The function is not differentiable but it is continuous -> we can use SGD

    - For full objective function: sample several corrupt windows per true one. Sum over all training windows. Similar to negative sampling in word2vec.

### Matrix calculus introduction

- Chain Rule
  - For one-variable functions: multiply derivatives
  - For multiple variables at once: multiply Jacobians
- Scalar to vector

$$
f(\pmb x)=f(x_1,x_2,...,x_n)
$$

$$
\frac{\part f}{\part {\pmb x}}= \left[\frac{\part f}{\part x_1},\frac{\part f}{\part x_2},...,\frac{\part f}{\part x_n}\right]
$$

- Scalar to matrix

$$
\pmb f(\pmb x)=\left[f_1(x_1,x_2,...,x_n),...,f_m(x_1,x_2,...,x_n) \right]
$$

​		Jacobian matrix
$$
\frac{\part\pmb f}{\part\pmb x}=\begin{bmatrix} 
\frac{\part f_1}{\part x_1} &\cdots &\frac{\part f_1}{\part x_1}\\
\vdots &\ddots &\vdots \\
\frac{\part f_m}{\part x_1} &\cdots &\frac{\part f_m}{\part x_n}\\
\end{bmatrix} \qquad

\left(\frac{\part\pmb f}{\part\pmb x} \right)_{ij} = \frac{\part f_i}{\part x_j}
$$

- An elementwise activation function applied a vector ( $\pmb z =f(\pmb x)$ )
  $$
  \left(\frac{\part\pmb z}{\part\pmb x} \right)_{ij} = \frac{\part z_i}{\part x_j}=\frac{\part }{\part x_j}f(x_i) = \begin{cases}
  f^\prime(x_i)  &\text{if } i=j\\
  0  &\text{if otherwise}
  \end{cases}
  $$

  $$
  \frac{\part\pmb z}{\part\pmb x}=\text{diag}(f^\prime(\pmb x))
  $$

- Other Jacobians
  $$
  \begin{align}
  &\frac{\part}{\part \pmb x}(\pmb W\pmb x+\pmb b)=\pmb W \\
  &\frac{\part}{\part \pmb b}(\pmb W\pmb x+\pmb b)=\pmb I \\
  &\frac{\part}{\part\pmb u}(\pmb u^T\pmb h)=\pmb h^T 
  \end{align}
  $$

- shape convention: **The shape of the gradient equals the shape of the parameter**

- What shape should dervatives be? Disagreement between Jacobian form (which makes the chain rule easy) and the shape convention (which makes implementing SGD easy)
  - Use Jacobian form as much as possible, reshape to follow the convention at the end
  - Always follow the convention (Look at dimensions to figure out when to transpose and/or reorder terms)