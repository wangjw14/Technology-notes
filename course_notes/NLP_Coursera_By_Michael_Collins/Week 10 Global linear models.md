# Week 10 Global linear models

- Supervised Learning in Natural Language

  - **General task:** induce a function $F$ from members of a set $\mathcal X$ to members of a set $\mathcal Y$ . e.g,

    | Problem             | $x\in\mathcal X$ | $y\in \mathcal Y$ |
    | ------------------- | ---------------- | ----------------- |
    | Parsing             | sentence         | Parse tree        |
    | Machine translation | French sentence  | English sentence  |
    | POS tagging         | sentence         | Sequence of tags  |

  - Supervised learning

    we have a training set $(x_i,y_i)$ for $i=1...n$

- History-based models

  - Break structures down into a derivation, or sequence of decisions

  - Each decision has an associated conditional probability

  - Probability of a structure is a product of decision probabilities

  - Parameter values are estimated using variants of maximum-likelihood estimation

  - Function $F:\mathcal X\rightarrow\mathcal Y$ is defined as 
    $$
    F(x)=\arg\max_yp(x,y;\Theta)\ \ \ \text{or} \ \ \ F(x)= \arg\max_yp(y|x;\Theta)
    $$

- Examples of history-based models
  - PCFGs
  - Log-linear taggers

### Global linear model as a framework

- Global linear model

  - move away from history-based models, no ideas of a "derivation"
  - we will have feature vectors over entire structure "Global features"

- Motivation of GLM:

  - Freedom in defining features
    - Parallelism in coordination
    - Semantic features

- Three components of global linear models

  - $\mathsf f$ is a function that maps a structure $(x,y)$ to a **feature vector** $\mathsf f(x,y)\in \R^d$

    - Feature: a function on a structure.

       $h(x,y)$ = Number of times $A\rightarrow B\ C$ is seen in $(x,y)$

    - Feature vector: a set of functions $h_1...h_d$ define a feature vector.  
      $$
      \mathsf f(x)=\langle h_1(x),h_2(x)...h_d(x) \rangle
      $$

  - GEN is a function that maps an input $x$ to a set of **candidates** GEN$(x)$

    - Some examples of how GEN($x$) can be defined
      - Parsing: set of parses for $x$ under a grammar
      - Any task: top N most probable parses under a history-based model
      - Tagging: set of all possible tag sequences with the same length as $x$
      - Translation: set of all possible English translations for the French sentence $x$

  - $\mathsf v$ is a parameter vector (also a member of $\R^d$)

  - Training data is used to set the value of $\mathsf v$

- Put components together

  - $\mathcal X$ is set of sentences, $\mathcal Y$ is set of possible outputs  

  - Need to learn a function $F:\mathcal X\rightarrow\mathcal Y$ 
    $$
    F(x) = \arg\max_{y\in \text{GEN}(x)} \mathsf f(x,y)\cdot \mathsf v
    $$

### Parsing problems in this framework (Reranking problems)

- Reranking approaches to parsing

  - Use a **baseline** parser to produce top $N$ parses for each sentence in training and test data, for example, use a lexicalized PCFG as GEN($x$) to generate a number of parses.
  - **Supervision:** for each $x_i$ take $y_i$ to be the parse that is "closest" to the treebank parse in GEN($x_i$)

- The representation of $\mathsf f$

  - Each component of $\mathsf f$ could be essentially **any** feature over parse tree

  - For example:

    $f_1(x,y)=$ log probability of $(x,y)$ under the baseline model

    $f_2(x,y)= \begin{cases} 1 &\text{if $(x,y)$ includes the rule $\rightarrow$ VP PP VBD NP}\\0 & \text{othrewise}\end{cases}$

  - Some long range rules are used as features in GLM, for example$\text{VP -> PP  VBD  NP  NP  SBAR}$

  - Bigrams in the long range rules, to the left and right of the head of the rule. For example, $\text{(Right, VP, NP, NP)}$, $\text{(Right, VP, NP, SBAR)}$, $\text{(Right, VP, SBAR, STOP)}$ and $\text{(Left, VP, PP, STOP)}$

  - Grandparent rules, which is rules including the non-terminal above the rule.

### A variant of the perceptron algorithm

| **Inputs:**         | Training set $(x_i,y_i)$ for $i=1...n$                       |
| ------------------- | ------------------------------------------------------------ |
| **Initialization:** | $\mathsf v=0$                                                |
| **Define:**         | $F(x) = \arg\max_{y\in \text{GEN}(x)} \mathsf f(x,y)\cdot \mathsf v$ |
| **Algorithm:**      | For $t=1...T,i=1...n \\ \ \ \ \ z_i=F(x_i)\\ \ \ \ \ \text{If} \ (z_i\ne y_i)\ \  \mathsf v = \mathsf v + \mathsf f(x_i,y_i) - \mathsf f(x_i,z_i)$ |
| **Output:**         | Parameters $\mathsf v$                                       |

- Performance

|                  | Collins 1999    | Charniak 2005  |
| ---------------- | --------------- | -------------- |
| Generative model | 88.2% F-measure | 89.7% accuracy |
| Reranked model   | 89.5% F-measure | 91.0% accuracy |

## The perceptron algorithm for tagging

- Tagging using Global Linear Models

  - Inputs $x$ are sentences $w_{[1:n]}=\{w_1...w_n\}$
  - Define $\mathcal T$ to be the set of possible tags
  - GEN$(w_{[1:n]})=\mathcal T^n$ i.e. all tag sequences of length $n$
  - Note: The size if GEN is exponential in the sentence length

- Local Feature-vector representations

  - Take a history/tag pair $(h,t)$ where $h_i=\langle t_{i-2},t_{i-1},w_{[1:n]},i \rangle$
  - $g_s(h,t)$ for $s=1...d$ are **local features** representing tagging decision $t$ in context $h$
  - A tagged sentence with $n$ words has $n$ history/tag pairs

- **Global features**

  - Define global features through local features

  $$
  \mathsf f(t_{[1:n]},w_{[1:n]}) = \sum_{i=1}^ng(h_i,t_i)
  $$

  - Typically, local features are indicator functions, and global features are the counts

- Put it all together

  - GEN$(w_{[1:n]})$ is the set of all tagged sequences of length $n$

  - GEN, $\mathsf {f,v}$ define
    $$
    \begin{align}
    F(w_{[1:n]})&= \arg\max_{t_{[1:n]}\in \text{GEN}(w_{[1:n]})} \mathsf v\cdot \mathsf f(w_{[1:n]},t_{[1:n]})\\
    &=\arg\max_{t_{[1:n]}\in \text{GEN}(w_{[1:n]})} \mathsf v\cdot \sum_{i=1}^ng(h_i,t_i))\\
    &=\arg\max_{t_{[1:n]}\in \text{GEN}(w_{[1:n]})} \sum_{i=1}^n\mathsf v\cdot g(h_i,t_i))
    \end{align}
    $$

  - Dynamic programming can be used to find the $\arg\max$

- Training a tagger using the perceptron algorithm 

  - **Inputs:** Training set $(w^i_{[1:n_i]},t^i_{[1:n_i]})$ for $i=1...n$.

  - **Initialization:** $\mathsf v=0$ 

  - **Algorithm:** For $t=1...T,i=1...n$
    $$
    z_{[1:n_i]}=\arg\max_{u_{[1:n_i]}\in\mathcal T^{n_i}} \mathsf v\cdot \mathsf f((w^i_{[1:n_i]},u^i_{[1:n_i]}))
    $$
    $z_{[1:n_i]}$ can be computed with the dynamic programming (Viterbi) algorithm.

    If $z_{[1:n_i]}\ne t^i_{[1:n_i]}$ then
    $$
    \mathsf v=\mathsf v+\mathsf f (w^i_{[1:n_i]},t^i_{[1:n_i]})-\mathsf f(w^i_{[1:n_i]},z_{[1:n_i]})
    $$

  - **Output:** Parameter vector $\mathsf v$.

- Experiments results:

  - Wall Street Journal part-of-speech tagging data

    Perceptron = 2.89% error, Log-linear tagger =3.28% error

  - NP chunking data

    Perceptron = 93.63% accuracy, Log-linear tagger = 93.29% accuracy

## The perceptron algorithm for dependency parsing

- Denpendency parsing

  - root is a special symbol
  - Each dependency is a pair $(h,m)$ where $h$ is the index of a head word, $m$ is the index of a modifier word
  - Conditions on dependency structures
    - The dependency arcs form a directed tree, with the root symbol at the root of the tree
    - There are no "crossing dependencies"
  - There are many resources for dependency parsing
  - Efficiency of dependency parsing
    - Define $n$ as the length of the sentence, $G$ is the number of non-terminals in the grammar
    - PCFG parsing is $O(n^3G^3)$ 
    - Lexicalized PCFG parsing is $O(n^5G^3)$
    - Unlabeled dependency parsing is $O(n^3)$ 

- GLMs for dependency parsing
  - The algorithm 

    - Local features

       $g(x,h,m)$ maps a sentence $x$ and a dependency $(h,m)$ to a local feature vector

    - Global features
      $$
      \mathsf f(x,y) = \sum_{(h,m)\in y} g(x,h,m)
      $$

    - Using dynamic programming to calculate:
      $$
      \begin{align}
      F(x)&=\arg\max_{y\in \text{GEN}(x)} \mathsf w\cdot \mathsf f(x,y)\\
      &=\arg\max_{y\in \text{GEN}(x)}\sum_{(h.m)\in y} \mathsf w\cdot g(x,h,m)
      \end{align}
      $$

  - Features from McDonald

    - Unigram features: Indentity of $w_h$. Indentity of $w_m$. Indentity of $t_h$. Indentity of $t_m$. 
    - Bigram features: Indentity of the 4-tuple $\lang w_h,w_m,t_h,t_m \rang$ or the subsets of this 4-tuple.
    - Contextual features: Indentity of the 4-tuple $\lang t_h,t_{h+1},t_{m-1},t_m \rang$.
    - In-between features: Indentity of triples $\lang t_h,t,t_m \rang$ for any tag $t$ seen between words $h$ and $m$.

- Results from McDonald (2005)

  | Method                           | Accuracy |
  | :------------------------------- | :------- |
  | Collins (1997) lexicalized PCFGs | 91.4%    |
  | 1st order dependency             | 90.7%    |
  | 2nd order dependency             | 91.5%    |

  - Advantages of the dependency parsing approaches: simplicity, efficiency ($O(n^3)$ parsing time)

## Summary

- Global linear model require definitions of GEN, $\mathsf f$
- Key ideas in tagging and dependency parsing
  - GEN is set of all possible structures (exponential in size)
  - $\mathsf f$ is defined through a sum of local feature vectors ($\mathsf f(x,y)=\sum g(...)$)
  - Dynamic programming is used to find the highest scoring structure
- We've seen the perceptron algorithm for parameter estimation, but there are other options:
  - Conditional random fields ("giant" log-linear models, gradient ascend for parameter estimation)
  - Large-margin methods (related to support vector machines)