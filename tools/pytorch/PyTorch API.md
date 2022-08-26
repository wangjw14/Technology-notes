# PyTorch API

## dataset

- dataset：The `Dataset` retrieves our dataset’s features and labels one sample at a time. 

- dataloader：While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s `multiprocessing` to speed up data retrieval.
  
  `DataLoader` is an iterable that abstracts this complexity for us in an easy API.

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

```python

```



## load model

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