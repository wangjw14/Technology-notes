# 使用Cython调用C++代码

- C++头文件（`inference_image.h`）

  ```c++
  #ifndef _INFERENCE_IMAGE_H
  #define _INFERENCE_IMAGE_H
  void print_image_func(int* v,int height, int width, int channel);
  #endif
  ```

- C++代码文件（`inference_image.cpp`）

  ```c++
  #include "inference_image.h"
  #include<iostream>
  #include<vector>
  
   std::vector<std::vector<std::vector<int> > > to3D(int* v,int height, int width, int channel){
   	std::vector<std::vector<std::vector<int> > > res;
   	int count =0;
   	for (int i=0; i<height;++i){
   		std::vector<std::vector<int> > row;
   		for (int j=0;j<width;++j){
   			std::vector<int> pixel;
   			for (int k=0;k<channel;++k){
   				pixel.push_back(v[count]);
   				count ++;
   			}
   			row.push_back(pixel);
   		}
   		res.push_back(row);
   	}
   	return res;
   }
  
   void print_image_func(int* v,int height, int width, int channel){
   	std::vector<std::vector<std::vector<int> > > v3d;
   	v3d = to3D(v,height,width,channel);
   	for (int i=0;i<height;++i){
   		for (int j=0;j<width;++j){
   			for (int k=0;k<channel;++k)
   				std::cout<<v3d[i][j][k] << ' ';
   			std::cout<<' ' <<std::endl;
   		}
   		std::cout<<' ' <<std::endl;
   	}
   }
  
  int main(int argc, char const *argv[])
  {
  	int v[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  	print_image_func(v,2,3,4);
  	return 0;
  }
  ```

- cython文件（`_inference_image.pyx`）

  ```python
  # distutils: language=c++
  import numpy as np
  cimport numpy as np
  
  cdef extern from "inference_image.cpp":
      pass
  cdef extern from "inference_image.h":
      void print_image_func(int* v,int height, int width, int channel)
  
  def print_image(np.ndarray[int, ndim=1, mode="c"] image not None,
  	h,w,c):
      print_image_func(<int*> np.PyArray_DATA(image), h, w, c)
  
  ```

- setup文件（`setup.py`）

  ```python
  from setuptools import setup,Extension
  from Cython.Build import cythonize
  import os
  import numpy
  os.environ["CC"] = "g++"
  os.environ["CXX"] = "g++"
  setup(ext_modules = cythonize(Extension("inference_module",
      sources=["_inference_image.pyx"])),
      include_dirs=[numpy.get_include()])
  ```

- 编译

  ```
  python setup.py build_ext --inplace
  ```

  https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html

