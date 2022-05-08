# PyTorch API

- 使用torch.load()加载模型参数时，提示"xxx.pt is a zip archive(did you mean to use torch.jit.load()?)"

> 可以看到在torch1.6版本中，对torch.save进行了更改.The 1.6 release of PyTorch switched torch.save to use a new zipfile-based file format. torch.load still retains the ability to load files in the old format. If for any reason you want torch.save to use the old format, pass the kwarg _use_new_zipfile_serialization=False.

解决方法：
```python
# 训练所有数据后，保存网络的参数
torch.save(model.state_dict(), model_cp,_use_new_zipfile_serialization=False)  
```
```python
#在torch 1.6版本中重新加载一下网络参数
model = MyNetwork().to(device) #实例化模型并加载到cpu货GPU中
model.load_state_dict(torch.load(model_cp))  #加载模型参数，model_cp为之前训练好的模型参数（zip格式）
#重新保存网络参数，此时注意改为非zip格式
torch.save(model.state_dict(), model_cp, _use_new_zipfile_serialization=False)
```


原文链接：https://blog.csdn.net/weixin_44769214/article/details/108188126

- 