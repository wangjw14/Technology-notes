# python

## pandas中apply的加速方法

```python
from pandarallel import pandarallel
pandarallel.initialize()

df['sim'] = df.parallel_apply(lambda row: np.dot(row['embedding1'], row['embedding2']),axis=1)

# 50s可以完成2.4亿次上面的计算
```







