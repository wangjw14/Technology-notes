# Lecture 9 Practical Tips for Final Projects

### Default Final Project

- Stanford Qusetion Answering Dataset
  - https://rajpurkar.github.io/SQuAD-explorer/
  - Attempting SQuAD 2.0 rather than SQuAD 1.1 (has unanswerable Qs)

- Goals of Final project
  - Lofty level: good to know something about how to do research
  - Prosaic level: 
    - baselines
    - benchmarks
    - evaluation
    - Error analysis
    - paper writing

- Learn to write Project Proposal 
  - Find a relevant research paper for your topic
  - Wirte a summary of that research paper and describe how you hope to use or adapt ideas from it and how you plan to extend or improve it in your final project work
  - Write a plan, including: relevant existing literature, models you will use/explore, data you will use (and how it is obtained), and how you will evaluate success

### Finding research topics

- Two basic starting points, for all of science:
  - [Nails] start with a problem of interest and find good/better ways to address it than are currently know/used
  - [Hammers] start with a technical approach of interest, and work out good ways to extend or improve it or new ways to apply it

- How to find an interseting place to start
  - ACL anthology for NLP papers: https://www.aclweb.org/anthology/
  - Major ML conference: NeurIPS, ICML, ICLR
  - Online preprint servers: http://arxiv.org
  - Arxiv Sanity Preserver by Andrej Karpathy http://www.arxiv-sanity.com/
  - State of the art and paper with code https://paperswithcode.com/sota

### Finding data

- Linguistic Data Consortium
  - https://catalog.ldc.upenn.edu/
  - Stanford licenses data https://linguistics.stanford.edu/resources/resources-corpora
  - Treebanks, named entities, coreference data, newswire, lots of speech with transcription, parallel MT data
- Machine translation
  - http://statmt.org/
- Dependency parsing
  - https://universaldependencies.org/
- Many more
  - Kaggle
  - research papers
  - list of datasets
    - https://machinelearningmastery.com/datasets-natural-language-processing/
    - https://github.com/niderhoff/nlp-datasets

### Doing your research example: Apply NNets to Task

1. Define Task: Example: Summarization

2. Define Dataset

   1. Search for academic datasets
      - They already have baselines
      - E.g.: Newsroom Summarization Dataset: https://summari.es/
   2. Define your own data (harder, need new baselines)
      - Allows connection to your research
      - A fresh problem provides fresh opportunities!
      - There are lots of neat websites which provide creative opportunities for new tasks. (Twitter, Blogs, News, etc.)

3. Dataset hugiene

   - Right at the beginning, seperate off devtest and test splits
   - A seperate **tuning set** to tune hyperparameters
   - A **dev set** to measure progress, and a **dev2 set** if you overfit the dev set
   - **Only at the end,** you evaluate on **test set**, use final test set extremely few times, ideally only once.

4. Define your metric(s)

   - Search online for well established metrics on this task
   - Summarization: Rouge (Recall-Oriented Understudy for Gisting Evaluation) which defines n-gram overlap to human summaries
   - Human evaluation is still much better for summarization; you may be able to do a small scale human eval

5. Establish a baseline

   - Implement the simplest model first (often logistic regression on unigrams and bigrams or averaging word vectors)
   - For summarization: See LEAD-3 baseline
   - Compute metrics on train AND dev
   - Analyze errors
   - If metrics are amazing and no errors: Done! Problem was too easy. Need to restart.

6. Implement existing neural net model

   - Compute metric on train and dev 
   - Analyze output and errors

7. Always be close to your data! (Except for the final test set!)

   - Visualize the dataset
   - Collect summary statistics
   - Look at errors
   - Analyze how different hyperparameters affect performance

8. Try out different models and model variants

   Aim to iterate quickly via having a good experimental setup

   - Fixed window neural model, Recurrent neural network, Recursive neural network, Convolutional neural network, Attention-based model, ...

- Training a gated RNN

  1. Use an LSTM or GRU: it makes your life so much simpler!
  2. Initialize recurrent matrices to be orthogonal
  3. Initialize other matrices with a sensible (small!) scale
  4.  Initialize forget gate bias to 1: default to remembering
  5. Use adaptive learning rate algorithms: Adam, AdaDelta, ...
  6. Clip the norm of the gradient: 1–5 seems to be a reasonable threshold when used together with Adam or AdaDelta.
  7. Either only dropout vertically or look into using Bayesian Dropout (Gal and Gahramani – not natively in PyTorch)
  8. Be patient! Optimization takes time

- Run your model on a large dataset

  - It should still score close to 100% on the training data after optimization

    - Otherwise, you probably want to consider a more powerful model

    - Overfitting to training data is not something to bescared of when

      doing deep learning

  - But, still, regularize your model until it doesn’t overfit on dev data (L2 norm and dropout)

### Review of gated neural sequence models

- Two ways to slove vanishing gradients problem: attention and shortcut connections.

- Shortcut connections
  $$
  \begin{aligned} f\left(h_{t-1}, x_{t}\right)=u_{t} \odot \tilde{h}_{t}+\left(1-u_{t}\right) \odot h_{t-1}  \end{aligned}
  $$

  - Candidate Update:     $\tilde{h}_{t}=\tanh \left(W\left[x_{t}\right]+U\left(r_{t} \odot h_{t-1}\right)+b\right) $ 
  - Reset gate:                   $r_{t}=\sigma\left(W_{r}\left[x_{t}\right]+U_{r} h_{t-1}+b_{r}\right)$ 
  - Update gate:                $u_{t}=\sigma\left(W_{u}\left[x_{t}\right]+U_{u} h_{t-1}+b_{u}\right)$ 

  - Update gates add some adaptive shortcut connections between the different time step hidden states. And reset gates prune unnecessary connections adaptively. (Add more non-linearity)
  - The direct, linear connections between $h_t$ and $h_{t-1}$ are the secret of solving the vanishing gradients problem. (just like ResNets)

### The large output vocabulary problem in NLG

- Softmax computation for natural language generation is expensive, due to the large vocabulary size
- Possible approaches for outputs
  - Use a modest size vocabulary, about 50k. <UNK> for words not in the vocabulary.
  - Hierarchical softmax: tree-structured vocabulary
  - Noise-contrastive estimation: binary classification (Like negative sampling? need to confirm)
  - Train on subset of the vocabulary at a time; test on a smart on the set of possible translations (Jean,Cho,Memisevic,Bengio.ACL2015)
  - Use attention to work out what you are translating, you can do something like dictionary lookup
  - Word pieces; char. models

### MT Evaluation

- Evaluation for machine translation

  - Manual (the best!?)
    - Adequacy and Fluency
    - Error categorization
    - Comparative ranking of translations
  - Testing in an application that uses MT as one sub-component
    - question answering from foreign language documents
  - Automatic metric
    - BLEU (Bilingual Evaluation Understudy)
    - Others like TER, METEOR, ...

- BLEU Evaluation Metric

  - BLEU is a weighted geometric mean, with a brevity penalty factor added.

  - BLEU4 formula
    $$
    \exp(0.5\times\log p_1+0.25\times\log p_2+0.125\times\log p_3+0.125\times\log p_4-\\
    \max(\text{words-in-reference/words-in-machine}-1,0))
    $$

    - Only works at corpus level (zeros kill it), and there is a smoothed variant for sentence-level

  - MT BLEU scores now approach those of human translations but their true quality remains far below human translations

- Coming up with automatic MT evaluations has become its own research field

  - There are many proposals: TER, METEOR, MaxSim, SEPIA, our own RTE-MT
  - TERpA is a representative good one that handles some word choice variation

- MT research requires some automatic metric to allow a rapid development and evaluation cycle

