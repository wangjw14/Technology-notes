# Week 6 Phrase-Based Translation

### Learning phrases from alignments

- Get started

  - First stage in training a phrase-based model is extraction of a *phrase-based (PB) lexicon*
  - A PB lexicon pairs strings in one language with strings in another language, e.g.,  $natural\ language\ processing \ \leftrightarrow \ $自然语言处理

- Problems of aligments

  - Alignments are often *noisy*
  - MANY-TO-ONE problem, we can have alignments where multiple foreign language words align to a single English word, while wen cannot have alignments where multiple English words are aligned to the same foreign word.

- Finding Alignment Matrices

  - train IBM model 2 for $p(f|e)$, and come up with most likely alignment for each $(e,f)$ pair
  - train IBM model 2 for $p(e|f)$, and come up with most likely alignment for each $(e,f)$ pair
  - take intersection of the two alignments as a starting point

- Heurisitcs for Growing Alignments

  - Only explore alignment in union of $p(f|e)$ and $p(e|f)$ alignments
  - Add one alignment point at a time
  - Only add alignment points which align a word that currently has no alignment
  - At first, restrict ourselves to alignment points that are "neighbors" ( adjacent or diagonal) of current alignment points
  - Consider other alignment points

- Extracting Phrases Pairs from Alignment Matrix

  - A phrase-pair consists of a sequence of English words, $e$, paired with a sequence of foreign words, $f$
  - A phrase-pair $(e,f)$ is consistent if:
    - there is at least one word in $e$ aligned to a word in $f$
    - there are no words in $f$ aligned to words outside $e$
    - there are no words in $e$ aligned to words outside $f$
  - Extract all consisent phrase pairs from the training example

- Probability for Phrase Pairs

  For any phrase pair $(f,e)$ extracted from the training data, we can calculate 
  $$
  t(f|e)=\frac{Count(f,e)}{Count(e)}
  $$
  

### A phrase-based model

- A translation is going to involve picking a sequence of phrases. Each choice is going to involve **a language modeling score, a phrase score, and a distortion score**. 
- Search for the translation with the highest score under the combination of these three parameter types.

### Decoding in phrase-based models

- Definitions of Phrase-based Models

  1. A phrase-based model consists of:
     - A trigram language model, with parameter $q(w|u,v)$

     - A phrase-based lexicon has a score $g(f,e)$

     $$
     g(f,e)=\log (\frac{\mathsf{Count}(f,e)}{\mathsf{Count}(e)})
     $$

     - A distortion parameter $\eta$ (typically negative)

  2. For any phrase $p$

     - $p$ has three components $s(p),t(p),e(p)$
     - For a particular input (source-language) sentence $x_1…x_n$, a phrase is a tuple $(s,t,e)$ , signifying that the subsequence $x_s…x_t$ in the source language can be translated as the target language string $e$.
     - $\mathcal{P}$ is the set of all phrases for a sentence.

  3. Definition of derivation

     - A derivation $y$ is a finite sequence of phrase $p_1,p2…p_L$ , where each $p_j$ for  $j\in \{1…L\}$ is a member of $\mathcal{P}$

     - The length $L$ can be any positive integer value

     - For any derivation $y$ we use $e(y)$ to refer to the underlying translation defined by $y$. E.g.,
       $$
       y= \mathsf{(1,3,we \ must \ also),(7,7,take),(4,5,this\ criticism),(6,6,seriously)}
       $$
       and
       $$
       e(y)=\mathsf{we\ must\ also\ take\ this\ criticism\ seriously }
       $$

  4. Definition of **Valid Derivations**

     - For an input sentence $x=x_1...x_n$ , we use $\mathcal{Y}(x)$ to refer to the set of valid derivations for $x$
   - $\mathcal{Y}(x)$ is the set of all finite length sequences of phrases $p_1...p_L$ such that
       - Each $p_k$ for $k\in \{1...L\}$ is a member fo the set of phrases $\mathcal{P}$ for $x_1...x_n$
     - Each word in $x$ is translated exactly once
       - For all $k\in \{1...(L-1)\}$ , $|t(p_k)+1-s(p_{k+1})|\le d $  where $d\ge 0$ is a parameter of the model. In addition, we must have $|1-s(p_1)|\le d$

  5. Scoring Derivations

     - The optimal translation under the model for a source-language sentence $x$ will be
     $$
       \arg \max_{y\in\mathcal{Y}(x)} f(y)
     $$
  
   - In phrase-based systems, the score for any derivation $y$ is calculated as follows:
       $$
     f(y) = h(e(y)) +\sum_{k=1}^Lg(p_k) +\sum_{k=0}^{L-1}\eta\times|t(p_k)=1-s(p_{k+1})|
       $$
     where $h(e(y))$ is the trigram language model score. $g(p_k)$ is the phrase-based score for $p_k$. Parameter $\eta$ is the distortion penalty (typically negative). And we define $t(p_0)=0$
  
- Some definitions for Decoding Algorithm 

  - State tuple

    - A state tuple 
      $$
      (e_1,e_2,b,r,\alpha)
      $$
      where $e_1,e_2$ are last two English words, $b$ is a bit-string of length $n$. $r$ is an integer specifying the end-point of the last phrase in the state, and $\alpha$ is the score for the state.

    - The initial state is 
      $$
      q_0=(*,*,0^n,0,0)
      $$
      where $0^n$ is bit-string of length $n$, with $n$ zeros.

  - Transitions

    - we have $ph(q)$ for any state $q$, which returns set of phrases that are allowed to follow state $q=(e_1,e_2,b,r,\alpha)$.
    - For a phrase $p$ to be a member of $ph(q)$ , it must satisfy the following conditions:
      1. $p$ must not overlap with the bit-string $b$. We need $b_i=0$ for $i\in\{s(p)...t(p)\}$.
      2. The distortion limit must not be violated. More formally, we must have $|r+1-s(p)|\le d$ where $d$ is the distortion limit.

  - The $next$ function

    If $q=(e_1,e_2,b,r,\alpha)$ , and $p=(s,t,\epsilon_1...\epsilon_M)$, then $ next(q,p)$ is the state $q^{\prime}=(e_1^{\prime},e_2^{\prime},b^{\prime},r^{\prime},\alpha^{\prime})$ defined as follows:

    - First, define $\epsilon_{-1}=e_1$, and $\epsilon_0=e_2$.

    - Define $e_1^{\prime}=\epsilon_{M-1},e_2^{\prime}=\epsilon_{M}$

    - Define $b_i^{\prime}=1$ for $i\in\{s...t\}$ , define $b_i^{\prime}=b_i$ for $i\notin\{s...t\}$

    - Define $r^{\prime}=t$

    - Define
      $$
      \alpha^{\prime}=\alpha+g(p)+\sum_{i=1}^M \log q(\epsilon_i|\epsilon_{i-2},\epsilon_{i-1})+\eta\times|r+1-s|
      $$

  - The Equality Function

    - The function 
      $$
      \mathsf{eq}(q,q^{\prime})
      $$
      returns  True or False.

    - Assuming $q=(e_1,e_2,b,r,\alpha)$ and $q^{\prime}=(e_1^{\prime},e_2^{\prime},b^{\prime},r^{\prime},\alpha^{\prime})$, then $\mathsf{eq}(q,q^{\prime})$ is True if and only if $e_1=e_1^{\prime},e_2=e_2^{\prime},b=b^{\prime},r=r^{\prime}$.

  

  - Add $(Q,q^{\prime},q,p)$
    - If there is some $q^{\prime\prime}\in Q$ such that $eq(q^{\prime\prime},q^{\prime})=$ True:
      - If $\alpha(q^{\prime})>\alpha(q^{\prime\prime})$
        - $Q=\{q^{\prime}\} \cup Q \setminus \{q^{\prime\prime}\}$
        - set $bp(q^\prime)=(q,p)$
      - Else return
    - Else
      - $Q=Q\cup\{q^\prime\}$ 
      - set $bp(q^\prime)=(q,p)$

  - Beam$(Q)$

    - Define 
      $$
      \alpha^*=\max_{q\in Q} \alpha(q)
      $$
      i.e., $\alpha^*$ is the highest score for any state in $Q$

    - Define $\beta\ge0$ to be the beam-width parametr, Then 

    $$
    beam(Q)=\{q\in Q:\alpha(q)\ge \alpha^*-\beta\}
    $$

- The Decoding Algorithm

  - **Input: ** sentence $x_1...x_n$. Phrase-based model $(\mathcal{L},h,d,\eta)$ ( $\mathcal{L}$ is lexicon, $h$ is language model, $d$ is distortion limit, and $\eta$ is the distortion parameter.) The phrase-based model defines the functions $ph(q)$ and $next(q,p)$

  - **Initialization:** set $Q_0=\{q_0\},Q_i=\emptyset$ for $i=1...n$.

  - For $i=0...(n-1)$

    - For each state $q\in beam(Q_i)$, for each phrase $p\in ph(q)$:

      (1) $q^\prime=next(q,p)$

      (2) Add $(Q,q^{\prime},q,p)$ where $i=len(q^\prime)$

  - **Return:** highest scoring state in $Q_n$. Backpointers can be used to find the underlying sequence of phrases (and the translation).

  
