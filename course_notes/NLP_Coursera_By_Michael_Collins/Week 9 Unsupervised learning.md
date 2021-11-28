# Week 9 Unsupervised learning

#### The Brown Clustering Algorithm

- Overview of the algorithm
  - **Input:** a (large) corpus of words
  - **Output:** a partition of words into word clusters (hierarchichal)

- The intuition

  - Similar words appear in similar contexts
  - More precisely: similar words have similar distributions of words to their immediate left and right

- The Formulation

  - **Vocabulary:** $\mathcal V$ is the set of all words seen in the corpus $w_1,w_2,...,w_T$

  - **Function:** Say $C:\mathcal V\rightarrow\{1,2...k\}$ is a partition of the vocabulary into $k$ classes

  - The model
    $$
    p(w_1,w_2,...w_T)=\prod_{i=1}^ne(w_i|C(w_i))q(C(w_i)|C(w_{i-1}))
    $$
    note: $C(w_0)$ is a special start state

  - More conveniently:
    $$
    \log p(w_1,w_2,...w_T)=\sum_{i=1}^n \log e(w_i|C(w_i))q(C(w_i)|C(w_{i-1}))
    $$

  - **Parameters:** 
    $$
    \begin{align}
    &\text{emission parameter:} \ \ \ e(v|c) \ \ \    \text{for}\ \  v\in \mathcal V , c\in\{1...k\} \ \   \\
    &\text{transition parameter:} \ \ \ q(c^\prime|c) \ \ \    \text{for}\ \  c^\prime, c\in\{1...k\} \ \ 
    \end{align}
    $$

- The difference with HMM

  - The function $C$ is deterministic, that each word gets mapped to a single state

- Measure the quality of $C$

  - How to measure the quality of a partition of $C$
    $$
    \begin{align}
    \text{Quality}(C) &= \sum_{i=1}^n \log e(w_i|C(w_i))q(C(w_i)|C(w_{i-1}))\\
    &= \sum_{c=1}^k\sum_{c^\prime=1}^k p(c,c^\prime)\log\frac{p(c,c^\prime)}{p(c),p(c^\prime)}+G
    \end{align}
    $$
    where $G$ is a constant

  - Here
    $$
    p(c,c^\prime)=\frac{n(c,c^\prime)}{\sum_{c,c^\prime}n(c,c^\prime)}
    $$

    $$
    p(c)=\frac{n(c)}{\sum_cn(c)}
    $$

    where $n(c)$ is the number of times class $c$ occurs in the corpus, $n(c,c^\prime)$ is the number of times $c^\prime$ seen following $c$, under the function $C$

- Algorithm to find $C$
  - Parameter of the approach is $m$ (e.g., $m$=1000)
  - Take the top $m$ most frequent words, put each into its own cluster, $c_1,c_2,...c_m$
  - For $i=(m+1)...|\mathcal V|$
    - Create a new cluster $c_{m+1}$, for the $i^\prime$ th most frequent word. We now have $m+1$ clusters
    - Choose two clusters from $c_1...c_{m+1}$ to be merged: pick the merge that gives a maximum value for Quality(C). We're now back to $m$ clusters.
  - Carry out $(m-1)$ final merges, to create a full hierarchy.
  - **Runtime:** $O(|\mathcal V|m^2+n)$ where $n$ is corpus length

-  Name tagging with word clusters and discriminative training
  - Use word clusters and log-linear model to make a name tagger.
  - Add features that look at the bit strings corresponding to the words in the context.
  - Using features that have information of word clusters can reduce the amount of train data and imporve the performance of the algorithm.