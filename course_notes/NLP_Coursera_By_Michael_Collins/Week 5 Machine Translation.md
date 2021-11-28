# Week 5 Machine Translation

### Challenges in machine translation

- Lexical ambiguity
- Differing word orders
- Syntactic structure is not preserved across translations
- Syntactic ambiguity causes problems
- Pronoun resolution

### Classical machine translation

- Direct Machie traslation
  - Definitions
    - Traslation is word-by-word
    - Very little analysis of the source text (e.g., no syntactic or sematic analysis)
    - Relies on a large bilingual directioanry. For each word in the source language, the directory specifies a set of rules for translating that word
    - After the words are translated, simple reorderig rules are applied.
  - Problem of direction translation
    - Lack of any analysis of the source language 
    - Different or impossible to capture long-range reordering
    - Words are translated without disambiguation of their syntactic role
- Transfer-based Approaches 
  - Three phases in translation
    - **Analysis:** Analyze the source language sentence; for example, build a syntactic analysis of the source language sentence.
    - **Transfer:** Covert the source-language parse tree to a target-language parse tree.
    - **Generation:** Covert the target-language parse tree to an output sentence.
  - Properties
    - The "parse trees" involved can vary from shallow analyses to much deeper analyses (even semantic representations)
    - The transfer rules might look quite similar to the rules for direct translation systems. But they can now operate on syntactic structures.
    - It is easier to handle long-distance reordering.
    - The *Systran* systems are a classic example of this approach.
- Interlingua-Based Translation
  - Two phases in translation
    - **Analysis:** Analyse the source language sentence into a (language-independent) representation of its meaning
    - **Generation:** Convert the meaning representation into an output sentence.
  - Advantage: A translation system that translates between $n$ languages, just need $n$ analysis components and $n$ generate components, while a transfer rules based system needs $O(n^2)$ sets of translation rules.
  - Disadvantage: It is difficult to get a language-independent representation
    - Different languages break down concepts in quite different ways
    - An interlingua might end up simple being an intersection of these different ways of breaking down concepts, but that doesn't seem very satisfactory. 

### A brief introduction to statistical MT

- Some introduction

  - Parallel corpora are available in several language pairs
  - Basic idea: use a parallel corpus as a training set of translation examples
  - Idea back to Warren Weaver (1949): suggested applying statistical and cryptanalytic techniques to translation.
  - Classic example: IBM work on French-English translation

- The Noisy Channel Model

  - Goal: translation system from French to English

  - Have a model $p(e|f)$ which estimates conditional probability of any English $e$ given the French sentence $f$. 

  - Two components of the model:

    $p(e)$   **the language model**, could be a trigram model, estimated from any data

    $p(f|e)$  **the translation model**, trained from a parallel corpus

  - Giving:
    $$
    p(e|f) = \frac{p(e,f)}{p(f)} = \frac{p(e)p(f|e)}{\sum_ep(e)p(f|e)}
    $$
    and
    $$
    \arg \max_ep(e|f) = \arg \max_ep(e)p(f|e)
    $$

  - Note:

    - The translation model is backwards (Generative model)
    - The language model can make up for deficiencies of the translation model
    - Decoding, i.e., finding $\arg \max_ep(e)p(f|e)$ is also a challenging problem



## The IBM Translation Models

### IBM Model 1

- Alignments
  - English sentence $e$ has $l$ words $e_1…e_l$ , French sentence $f$ has $m$ words $f1…f_m$
  - An alignments $a$ identifies which English word each French word originated from. An alignment $a$ is $\{a1,…,a_m\}$ , where each $a_j\in \{0…l\}$
  - There are $(l+1)^m$ possible alignments.

- Alignments in IBM models

  - Define models for $p(a|e,m)$ and $p(f|a,e,m)$, giving

  $$
  p(f,a|e,m) = p(a|e,m)p(f|a,e,m)
  $$

  ​		and 
  $$
  p(f|e,m)=\sum_{a\in \mathcal{A}}p(f,a|e,m)=\sum_{a\in \mathcal{A}}p(a|e,m)p(f|a,e,m)
  $$
  ​		where $\mathcal{A}$ is the set of all possible alignments.

  - we can also get 

  $$
  p(a|f,e,m) = \frac{p(f,a|e,m)}{\sum_{a\in\mathcal{A}}p(f,a|e,m)}
  $$
  ​		For a given $f,e$ pair, we can also compute the most likely alignment
  $$
  a^*=\arg \max_a p(a|f,e,m)
  $$

- **IBM Model 1**

  - In IBM model 1 all alignments $a$ are equally likely

  $$
  p(a|e,m) = \frac{1}{(l+1)^m}
  $$

  ​		this is a major simplifying assumption, but it gets things started.

  - 

  - $$
    p(f|a,e,m)=\prod_{j=1}^mt(f_j|e_{a_j})
    $$

  - **To generate a French string $f$ from a English string $e$ **

    - **Step 1:** Pick an alignment $a$ with probability $\frac{1}{(l+1)^m}$
    - **Step 2:** Pick the French words with probability 

    $$
    p(f|a,e,m)=\prod_{j=1}^mt(f_j|e_{a_j})
    $$

    **The final result:**
    $$
    p(f,a|e,m)= p(a|e,m)\times p(f|a,e,m)=\frac{1}{(l+1)^m}\prod_{j=1}^mt(f_j|e_{a_j})
    $$

### IBM Model 2

- Only difference with model 1: introduce distortion parameters for alignment.

  Define $q(i|j,l,m)$ as probability that $j$ th French word is connected to $i$ th English word, given sentence lengths of $e$ and $f$ are $l$ and $m$ respectively.

- Define 
  $$
  p(a|e,m)= \prod_{j=1}^mq(a_j|j,l,m)
  $$
  Then
  $$
  p(f,a|e,m)=\prod_{j=1}^mq(a_j|j,l,m)t(f_j|e_{a_j})
  $$

- **To generate a French string $f$ from an English string $e$ **

  - **Step 1:** Pick an alignment $a=\{a_1,a_2…a_m\}$ with probability

  $$
  \prod_{j=1}^mq(a_j|j,l,m)
  $$

  - **Step 2:** Pick the French words with probability

  $$
  p(f|a,e,m)=\prod_{j=1}^mt(f_j|e_{a_j})
  $$

  - **The final result:**

  $$
  p(f,a|e,m)= p(a|e,m)\times p(f|a,e,m)=\prod_{j=1}^mq(a_j|j,l,m)t(f_j|e_{a_j})
  $$

- Recovering Alignments

  - If we have parameters $q$ and $t$, we can easily recover the most likely alignment for any sentence pair
  - Given a sentence $e_1,e_2,…,e_l,f_1,f_2,…,f_m$ , define

  $$
  a_j=\arg \max_{a\in\{0...l\}} q(a|j,l,m)\times t(f_j|e_a)
  $$

  ​		for $j=1…m$

### EM training of Models 1 and 2

- The parameter estimation problem

  - **Input:** $(e^{(k)},f^{(k)})$ for $k=1…n$. Each $e^{(k)}$ is a English sentence, each $f^{(k)}$ is a French sentence
  - **Output:** parameters $t(f|e)$ and $q(i|j,l,m)$ 

- Parameter Estimation with Alignments are observed

  - **Input:** 

    A training corpus $(f^{(k)},e^{(k)},a^{(k)})$ for $k=1…n$, where $f^{(k)}=f^{(k)}_1…f^{(k)}_{m_k},e^{(k)}=e^{(k)}_1…e^{(k)}_{l_k},a^{(k)}=a^{(k)}_1…a^{(k)}_{m_k}$

  - **Algorithm:** 

    - Set all counts $c(…)=0$

    - For $k=1…n$

      - For $i=1…m_k$, For $j=0…l_k$,
        $$
        \begin{align}
        c(e^{(k)}_j,f^{(k)}_i)\ &\leftarrow \ c(e^{(k)}_j,f^{(k)}_i) +\delta(k,i,j) \\
         c(e^{(k)}_j)\ &\leftarrow \ c(e^{(k)}_j) +\delta(k,i,j) \\
        c(j|i,l,m)\ &\leftarrow \ c(j|i,l,m) +\delta(k,i,j) \\
        c(i,l,m)\ &\leftarrow \ c(i,l,m) +\delta(k,i,j) \\
        \end{align}
        $$
        where $\delta(k,i,j)=1$ if $a^{(k)}_i=j, \ 0$ otherwise.

  - **Output:** 
    $$
    t_{ML}(f|e)=\frac{c(e,f)}{c(e)} \\q_{ML}(j|i,l,m)=\frac{c(j|i,l,m)}{c(i,l,m)}
    $$

- Parameter Estimation with the EM Algorithm

  - **Input:** 

    A training corpus $(f^{(k)},e^{(k)})$ for $k=1…n$, where $f^{(k)}=f^{(k)}_1…f^{(k)}_{m_k},e^{(k)}=e^{(k)}_1…e^{(k)}_{l_k}$ (No alignments now)

  - **Initialization:**

    Initialize $t(f|e)$ and $q(j|i,l,m)$ parameters (e.g., to random values)

  - **Algorithm:** 

    - For $s=1…S$  (iterate for several times, generally 10-20 times)
      - Set all counts $c(…)=0$

      - For $k=1…n$

        - For $i=1…m_k$, For $j=0…l_k$,
          $$
          \begin{align}
          c(e^{(k)}_j,f^{(k)}_i)\ &\leftarrow \ c(e^{(k)}_j,f^{(k)}_i) +\delta(k,i,j) \\
           c(e^{(k)}_j)\ &\leftarrow \ c(e^{(k)}_j) +\delta(k,i,j) \\
          c(j|i,l,m)\ &\leftarrow \ c(j|i,l,m) +\delta(k,i,j) \\
          c(i,l,m)\ &\leftarrow \ c(i,l,m) +\delta(k,i,j) \\
          \end{align}
          $$
          where 
          $$
          \delta(k,i,j) = \frac{q(j|i,l_k,m_k)t(f^{(k)}_i|e^{(k)}_j)}{\sum_{j=0}^{l_k}q(j|i,l_k,m_k)t(f^{(k)}_i|e^{(k)}_j)}
          $$

      - Recalculate the parameters:
        $$
        t(f|e)=\frac{c(e,f)}{c(e)} \\q(j|i,l,m)=\frac{c(j|i,l,m)}{c(i,l,m)}
        $$

- Justification for the algorithm

  - The log-likelihood function

  $$
  L(t,q)=\sum_{k=1}^n\log p(f^{(k)}|e^{(k)})= \sum^n_{k=1}\log\sum_ap(f^{(k)},a|e^{(k)})
  $$

  - The maximum-likelihood estimates are

  $$
  \arg \max_{t,q} L(t,q)
  $$

  