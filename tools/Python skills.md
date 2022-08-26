# Python skills

## 获取当前文件的路径

```python
cur_dir = os.path.abspath(os.path.dirname(__file__))
```

## 设置参数

```python
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, default='page_B')
parser.add_argument('--input', type=str)

args = parser.parse_args()
```

## 标准输入改变编码

- python2
  
  ```python
  sys.stdin = codecs.getreader('gb18030')(sys.stdin)
  sys.stdout = codecs.getwriter('gb18030')(sys.stdout)
  sys.stderr = codecs.getwriter('gb18030')(sys.stderr)
  ```

- python3
  
  ```python
  import io
  sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 
  sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') 
  ```

## SimpleHTTPServer

- python2
  
  ```python
  python -m SimpleHttpServer 8000
  ```

- python3
  
  ```python
  python3 -m http.server 8000
  
  # ocr原始数据，格式bos、ocr，\t分隔
  http://10.255.120.19:8011/shuaku/ocr_vis_res/top1_batch2_5w
  # asr原始数据，格式bos、asr，\t分隔
  http://10.255.120.19:8011/shuaku/asr_res/top1_batch2_5w
  # 根据ocr、asr生成的字幕，格式bos、subtitle，\t分隔
  http://10.255.120.19:8011/shuaku/data_bos_subtitle/top1_batch2_5w
  # tera信息，每行一个json串
  http://10.255.120.19:8011/shuaku/tera_data/top1_batch2_5w.json
  ```

## pytorch不同版本加载模型问题

https://blog.csdn.net/lbj1260200629/article/details/109848137

### Itertool functions

- combinations

```python
'''
itertools.combinations(iterable, r)
Return r length subsequences of elements from the input iterable
'''
from itertools import combinations
for i in combinations('ABCD',2):
    print(i)
```

results:

```python
('A', 'B')
('A', 'C')
('A', 'D')
('B', 'C')
('B', 'D')
('C', 'D')
```

- chain

```python
'''
chain(p, q,...),  return p0, p1, … plast, q0, q1, …
chain('ABC', 'DEF') --> A B C D E F
'''
from itertools import combinations,chain
for i in chain(['A',"B",1],[3,"H",2]):
    print(i)
```

results:

```
A B 1 3 H 2
```

- get all subsets

```python
from itertools import chain, combinations

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i in range(len(arr))])

for i in subsets('ABC'):
    print(i)
```

return

```python
('A',)
('B',)
('C',)
('A', 'B')
('A', 'C')
('B', 'C')
('A', 'B', 'C')
```

### Numpy one-hot embedding

```python
def one_hot(t, class_num):
    I = np.eye(class_num).astype(np.long)
    res = I[t]   
    return res
```

## list 维度转换

```python
bert_inputs, grid_labels, grid_mask2d = map(list, zip(*data))
```
