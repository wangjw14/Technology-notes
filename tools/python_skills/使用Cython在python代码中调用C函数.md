# 使用Cython在python代码中调用C函数

### Numpy支持

一共4段代码，分别如下：

generate_dist_id.h

```c
#ifndef _GENERATE_DIST_ID_H
#define _GENERATE_DIST_ID_H
void generate_dist_id(double * jd, double * wd, int * district_id,double * polygon_x, double * polygon_y,int data_len,int polygon_len,int idx);
#endif
```

generate_dist_id.c

```c
#include "generate_dist_id.h"
/*
判断一个点是否在一个多边形的内部
*/
int pnpoly(int npol, double *xp, double *yp, double x, double y)  
    {  
      int i, j, c = 0;  
      for (i = 0, j = npol-1; i < npol; j = i++) {  
        if ((((yp[i] <= y) && (y < yp[j])) ||  
             ((yp[j] <= y) && (y < yp[i]))) &&  
            (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))  
          c = !c;  
      }  
      return c;  
    }  

void generate_dist_id(double * jd, double * wd, int * district_id,
		double * polygon_x, double * polygon_y,int data_len,int polygon_len,int idx){
	int i = 0;
	for (i =0; i < data_len; i++){
		int result = pnpoly(polygon_len, polygon_x, polygon_y,jd[i],wd[i]);
		if (result)
			district_id[i] = idx;
	}
}
```

_generate_dist_id.pyx

```python
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "generate_dist_id.h":
	void generate_dist_id(double * jd, double * wd, int * district_id,
		double * polygon_x, double * polygon_y,int data_len,int polygon_len,int idx)

# create the wrapper code, with numpy type annotations
def generate_dist_id_func(np.ndarray[double, ndim=1, mode="c"] jd not None,
	np.ndarray[double, ndim=1, mode="c"] wd not None,
	np.ndarray[int, ndim=1, mode="c"] district_id not None,
	np.ndarray[double, ndim=1, mode="c"] polygon_x not None,
	np.ndarray[double, ndim=1, mode="c"] polygon_y not None,
	idx):
	generate_dist_id(<double*> np.PyArray_DATA(jd),
		<double*> np.PyArray_DATA(wd),
		<int*> np.PyArray_DATA(district_id),
		<double*> np.PyArray_DATA(polygon_x),
		<double*> np.PyArray_DATA(polygon_y),
		jd.shape[0],
		polygon_x.shape[0],
		idx)
```

setup.py

```python
from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext
setup(
   cmdclass={'build_ext': build_ext},
   ext_modules=[Extension("generate_dist_id",
                sources=["_generate_dist_id.pyx", "generate_dist_id.c"],
                include_dirs=[numpy.get_include()])],
)
```

在命令行运行

```shell
python setup.py build_ext -i
```

### 注意事项

numpy数组要注意C_CONTIGUOUS 为True

```python
jd = jd.copy(order='C')          # 设置array的flags
print(jd.flags)                  # 查看array的flags
```



# Reference

https://segmentfault.com/a/1190000000479951

https://stackoverflow.com/questions/26778079/valueerror-ndarray-is-not-c-contiguous-in-cython

