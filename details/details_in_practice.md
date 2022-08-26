#### warm up

- 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳

- 有助于保持模型深层的稳定性

- 一般可取训练steps的10%，参考BERT。

- 参考资料
  
  - [What does “learning rate warm-up” mean?](https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean)  
  - [神经网络中 warmup 策略为什么有效；有什么理论解释么？](https://www.zhihu.com/question/338066667) 



- dataloader与GPU利用率较低
  
  - https://www.daimajiaoliu.com/daima/8c7f749a5601c06