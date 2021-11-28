# Association rules

- The **support** of an item (itemset) $X$ is the percentage of transactions in which that item (itemset) occurs. 
  $$
  Support(X) = \frac{\#X}{n}
  $$

- The **support** of an association rule $X\rightarrow Y$ is the percentage of transactions that contain $X$ and $Y$
  $$
  Support (X\rightarrow Y)=\frac{\#(X\cup Y)}{n}
  $$

- The **confidence** of an association rule $X\rightarrow Y$ is the ratio of the number of transactions that contain $\{X,Y\}$ to the number of transcations that contain $X$
  $$
  \begin{align}
  Confidence(X\rightarrow Y)&= P(Y|X) \ \ \ \ \ \ \ \ \ \ \ \ \ \text{(Conditional probability)}  \\ \\
  &=\frac{Support(X\cup Y)}{Support X}\\ \\
  &=\frac{\#(X\cup Y)}{\# X} \\
  \end{align}
  $$

- Support and Confidence are bounded by thresholds:
  - Minimum support $\sigma$
  - Minimum confidence $\Phi$
  - A **frequent itemset** is an itemset with support larger than $\sigma$
  - A **strong rule** is a rule that is frequent and its confidence is higher than $\Phi$

- Association rules
  - Step 1: Find all frequent itemsets.
  - Step 2: Use frequent itemsets to generate association rules.
    - For each frequent itemset $f$
      - Create all non-empty subsets of $f$
    - For each non-empty subsets $s$ of $f$
      - Output $s\rightarrow (f-s)$ if $\frac{support(f)}{support(s)}>\Phi$

- Some mistakes (?)

  - A rule with high confidence is not necessarily plausible. (The confidence may lower than the priori probability)

    - For example, $|D|=10000,\#\{DVD\} =7500,\#\{Tape\}=6000,\\ \#\{DVD,Tape\}=4000, \sigma=0.3,\Phi=0.5$ 

      support(Tape$\rightarrow$DVD) = 0.4

      confidence(Tape$\rightarrow$DVD) =0.66<0.75

  - Association $\ne$ Causality

- The apriori method

  - Key ideas

    - A subset of a frequent itemset must be frequent.
    - The supersets of any infrequent itemset cannot be frequent.

  - Steps

    - Generate itemsets of a particular size, usually from size=1
    - Scan database once to see which of them are frequent
    - Use the frequent itemsets to generate candidate itemsets of size = size +1
    - Iteratively find frequent itemsets with cardinality from 1 to k
    - Avoiding generating candidates that are know to be infrequent

  - Algorithm

    - $C_k$: Candidate itemset of size k
    - $L_k$: Frequent itemset of size k
    - $L_1\leftarrow\{frequent \ items\}$
    - For $k=1;L_k\ne \empty;k++$
      - $C_{k+1}\leftarrow candidate(L_k)$
      - For each transaction t
        - $Q\leftarrow\{c|c\in C_{k+1},c\subseteq t\}$
        - count{c} $\leftarrow$ count{c}+1, $\forall c \in Q$
      - End For
      - $L_{k+1}\leftarrow \{c| c\in C_{k+1}, count(c)/N\ge\sigma\}$
    - End for

  - $L_k\rightarrow C_{k+1}$
    $$
    \{X\cup Y_k|\ \  X,Y\in L_k,X_i=Y_i\ \forall i \in[1,k-1],X_k\ne Y_k\}
    $$
    where $X,Y$ are ordered list.