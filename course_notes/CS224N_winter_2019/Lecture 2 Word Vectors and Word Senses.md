# Lecture 2: Word Vectors and Word Senses

### Count-based method

- Capture the information by co-occurance matrix
  - Two options: window vs full document
  - Window: similar to word2vec, captures both syntactic (POS) and semantic information
  - Word-document co-occurance matrix will give general tipics, leading to "Latent Semnatic Analysis" 
- Window based co-occurance matrix
  - Increase in size with vocabulary ( $|V|^2$ parameters)
  - Very high dimensional, require a lot of storage
  - Subsequent classification models have sparsity issues
  - Models are less robust
- Singular Value Decomposition of co-occurrence matrix $X$
  - Factorizes $X$ into $U\Sigma V^T$ where $U$ and $V$ are orthonormal.
  - Retain only $k$ singular values, in order to generalize.
  - $\hat X$ is the best rank $k$ approximation to $X$, in terms of least squares
- Some hacks to co-occurance matrix $X$
  - Scaling the counts in the cells can help a lot
  - Some fixes to high-frequency words:
    - $\min(X,t)$, with $t\approx 100$
    - Ignore them all
  - Ramped windows that count closer words more
  - Use Pearson correlations instead of counts, then set negative values to 0
- Count based vs. direct prediction
  - Count based: **LSA, HAL, COALS, Hellinger-PCA**
    - Pros: Fast training, efficient usage of statistics
    - Cons: Primarily used to capture word similarity, disproportionate importance given to large counts
  - Direct Prediction: **Skip-gram/CBOW, NNLM, HLBL, RNN**
    - Pros: Generate improved performance on other tasks, can capture complex patterns beyond words similarity
    - Cons: Scales with corpus size, Inefficient usage of statistics

### The GloVe model of word vectors

- Crucial insight: Ratios of co-occurrence probabilities can encode meaning components

- How can we capture ratios of co-occurrence probabilities as linear meaning components in a word vector space?

  - Log-bilinear model:
    $$
    w_i\cdot w_j=\log P(i|j)
    $$

  - With vector differences:
    $$
    w_x\cdot(w_a-w_b)=\log\frac{P(x|a)}{P(x|b)}
    $$

  - Cost function:
    $$
    J=\sum_{i,j=1}^Vf(X_{ij})(w_i^T\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2
    $$
    which means: dot product should be as similar as to the log of co-occurance probability. And $f$ is a ceiling function.

- GloVe unify two method above, which estimate simply of a count matrix, but it is done in the same kind of iterative loss based estimation method.

- Advantages of GloVe algorithm

  - Fast training
  - Scalable to huge corpora
  - Good performance even with small corpus and small vectors

### Evaluating word vectors

- Intrinsic vs extrinsic

  - Intrinsic:
    - Evaluation on a specific/intermdiate subtask
    - Fast to compute
    - Helps to understand that system
    - Not clear if really helpful unless correlation to real task is established
  - Extrinsic
    - Evaluation on a real task
    - Can take a long time to compute accuracy
    - Unclear if the subsystem is the problem or its interaction on other subsystems
    - If replacing exactly one subsystem with another imporves accuracy -> Winning!

- Intrinsic word vector evaluation

  - Word vector analogies

    Man:woman :: king:?
    $$
    d=\arg\max_i\frac{(x_b-x_a+x_c)^Tx_i}{||x_b-x_a+x_c||}
    $$

    - Dataset of analogies
      - Semantic examples: https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt
      - Syntactic examples: gram4-superlative
    - Good dimension for word vectors is about 300
    - Asymmetric context (only words to the left) are not as good
    - Window size of 8 around each center word is good for GloVe vectors
    - More data helps, Wikipedia is better than news text to buid word vectors

  - Word vector distances and their correlation with human judgments

    - Dataset: WordSim353 http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

- Extrinsic word vector evaluation

  - word embeddings can directly improve the profermance of many NLP tasks.

### Word sense ambiguity

- Many words have lots of meaning

  - Multiple word prototypes: cluster a word to several groups by context, then replace the original token by token_i, and run the word vectors algorithm to get several vevtors for one word.

  - Use weighted sum to build a standard word embedding like word2vec
    $$
    v_{\text{pike}}= \alpha_1v_{\text{pike}_1} +\alpha_2v_{\text{pike}_2} +\alpha_3v_{\text{pike}_3} 
    $$
    where $\alpha_1=\frac {f_1}{f_1+f_2+f_3}$ , etc, for frequency $f$

  - Because of ideas from **sparse coding**, you can actually separate out the different senses 



- 词向量是否经过归一化？