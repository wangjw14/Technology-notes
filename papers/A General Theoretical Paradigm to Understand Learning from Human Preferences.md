# A General Theoretical Paradigm to Understand Learning from Human Preferences

- 通过强化学习（RLHF）从人类偏好中学习的普遍部署依赖于两个重要的近似：第一个假设可以用pointwise代替pairwise。 第二个假设基于这些pointwise训练的奖励模型可以从收集的数据推广到策略采样的分布外数据。 最近，直接偏好优化（DPO）被提出作为一种绕过第二次近似并直接从收集的数据中学习策略的方法，而无需奖励建模阶段。 然而，该方法仍然严重依赖第一近似。

