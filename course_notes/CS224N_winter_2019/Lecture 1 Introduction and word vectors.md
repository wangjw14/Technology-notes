# Lecture 1 Introduction and word vectors

### The course

- course logistics
- what do we teach
  -  the key methods in NLP: Recurrent networks, attention, etc.
  - a big picture of human languages and the difficulties in understanding and producing them
  - build some systems (in PyTorch) for some major problems in NLP: word meaning dependency parsing, machine translation, question answering
- What is different this year
  - covering new material: character models, transformers, safety/fairness, multitask learn.

### Human language and word meaning

- word meaning 
  - **meaning** in dictionary: the idea
  - represent a word meaning in a computer
    - WordNet
    - Discrete symbols (one hot)
    - Word vectors
- thoughts about languages
  - language is uncertain evolved system of communication but somehow we have enough agreed meaning that we can kind of pretty much communicate.
  - human beings have knowledge that gives them intelligence, and human beings convey knowledge around world mainly by language
  - History of creatures' vision system is about 75 million years, while history of language is about 100 thousand years.  The history of writing is about 5 thousand years.
  - The development of languages made human beings invincible. It wasn't that human beings developed poison fangs or ability to run faster. Human have the unbeatable advantage that they could communicate with each other and therefore work much more effectively in teams.
  - Writing is the ability where you could take knowledge sent spatially around the world and then temporally through time. Writing was so powerful as a way of having knowledge that in those 5,000 years that enabled human beings to go from Stone Age to modern world.
  - Human language makes us a network. But human language is a pathetically slow network, and we don't get much bandwidth at all. 
  - Human came up with this incredibly impressive system which is essentially form of compression. When we talk to people, we assume that they have an enormous amount of knowledge in their heads which isn't the same but it's broadly similar. Therefore, I can say a short message and communicate only a relatively short bit string and you can actually understand a lot.

### Word2vec introduction 

- Representing words by their context

  - Distributional semantics: A word's meaning is given by the words that frequently appear close-by

- Word2vec

  - The idea of word2vec:

    - We have a large corpus of text
    - Every word in a fixed vocabulary is represented by a **vector**
    - Go through each position $t$ in the text, which has a center word $c$ and context words $o$
    - Use the similarity of the word vectors for $c$ and $o$ to calculate the probability of $o$ given $c$ (or vice versa)
    - Keep adjusting the word vectors to maximize the probability

  - objective function

    - For each position $t=1,...,T$  , predict context words with a window of fixed size $m$, given center word $w_j$

    $$
    \text{Likelihood}=L(\theta) = \prod_{t=1}^T\prod_{-m\le j\le m \\ \ \ \ \ \ j\ne 0}
    P(w_{t+j}|w_t;\theta)
    $$

    - Objective function
      $$
      J(\theta) = -\frac1T\log L(\theta)=-\frac1T\sum_{t=1}^T\sum_{-m\le j\le m \\ \ \ \ \ \ j\ne 0}\log P(w_{t+j}|w_t;\theta)
      $$

    - Use two vectors per word $w$, and $v_w$ for center word, $u_w$ for context word
      $$
      P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}
      $$
      where $V$ is the vocabulary

    - Softmax function maps arbitrarily values $x_i$ to a probability distribution $p_i$

      - "max" because amplifies probability of largest $x_i$
      - "soft" because still assigns some probability to smaller $x_i$ 

  - Parameters of the model: $\theta \in \Bbb R^{2dV}$ where we have $d$-dimensional vectors and $V$-many words. 

  - Why two vectors: Easier optimization. Average both at the end.

  - Two model variants:

    - Skip-grams: predict context words (position independent) given center word
    - Continuous Bag of words: predict center word from (bag of) context words

  - Additional efficiency in training:

    - **Negative sampling**: train binary logistic regressions for a true pair versus some noisy pairs.
      $$
      J(\theta) = \frac1T\sum_{t=1}^TJ_t(\theta)
      $$

      $$
      J_t(\theta)=\log \sigma(u_o^Tv_c)+ \sum_{i=1}^k\Bbb E_{j\sim P(w)}[\log \sigma(-u_j^Tv_c)]
      $$

      where 
      $$
      \sigma=\frac1{1+e^{-x}}
      $$

      $$
      P(w)=\frac{U(w)^{\frac3 4}}Z
      $$

      and $U(w)$ is the unigram distribution. 3/4 power can decrease the frequency of common words and increase the frequency of rare words. $Z$ is normalization term.

    - The form of $J(\theta)$ is taken from "Distributed Representations of Words and Phrases and their Compositionality” (Mikolov et al. 2013), and the function need to be maximized.

  - the model gives a reasonably high probability to all words in the context (don't contain position information)

  - High frequency problem: some words such as *and, the, that* may have high similarity socres with many other words.

- Gensim and word vector visualization

  - analogy
  - PCA and scatter plot

### Word2vec objective function gradients

$$
\frac{\part}{\part v_c}\log P(o|c)=u_o-\sum_{x=1}^vP(x|c)u_x
$$

### Optimization basics

- Gradient descent
  $$
  \theta^{new}=\theta^{old}-\alpha\nabla_{\theta} J(\theta)
  $$
  
- Stochastic gradient descent

  - Repeatedly sample windows, and update after each other
  - mini-batch of 32 or 64, make GPUs work faster and get less noise by averaging on mini-batch
  - as we use mini-batch, the gradients we get are sparse, therefore, we only update the certain rows of embedding matrices, g