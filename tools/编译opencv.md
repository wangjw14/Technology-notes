# 编译opencv

- 29%处

```shell
cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local CPU_BASELINE=SSE2 -DCPU_DISPATCH=SSE4_2,AVX ..
```

加入`CPU_BASELINE=SSE2 -DCPU_DISPATCH=SSE4_2,AVX ` 的作用是当前的cpu不支持AVX2:

参考链接：

https://github.com/opencv/opencv/issues/11120

https://github.com/opencv/opencv/wiki/CPU-optimizations-build-options

- 38%处

  grfmt_tiff.cpp:132:12: 错误：‘tmsize_t’不是一个类型名

  cc1plus: 警告：无法识别的命令行选项“-Wno-unnamed-type-template-args”

  ```shell
  cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local CPU_BASELINE=SSE2 -DCPU_DISPATCH=SSE4_2,AVX -D BUILD_TIFF=ON ..
  ```

  [https://anribras.github.io/tech/2018/02/09/python3-opencv3%E7%BC%96%E8%AF%91/](https://anribras.github.io/tech/2018/02/09/python3-opencv3编译/)

- 98%

  ```shell
  Traceback (most recent call last):
    File "/home/wangjingwen/work/opencv-master/modules/python/bindings/..//src2/gen2.py", line 4, in <module>
      import hdr_parser, sys, re, os
    File "/mnt/bigdata/work_2020/opencv-master/modules/python/src2/hdr_parser.py", line 914
      has_mat = len(list(filter(lambda x: x[0] in {"Mat", "vector_Mat"}, args))) > 0
                                                        ^
  SyntaxError: invalid syntax
  make[2]: *** [modules/python_bindings_generator/pyopencv_generated_include.h] 错误 1
  make[1]: *** [modules/python_bindings_generator/CMakeFiles/gen_opencv_python_source.dir/all] 错误 2
  make: *** [all] 错误 2
  ```

  ```shell
  cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local CPU_BASELINE=SSE2 -DCPU_DISPATCH=SSE4_2,AVX -D BUILD_TIFF=ON -D PYTHON_DEFAULT_EXECUTABLE=/opt/anaconda3/envs/tf113/bin/python  ..
  ```

  https://github.com/opencv/opencv/issues/7967





### 原始网页，基本步骤正确，具体的cmake命令参考上面的命令

https://www.vultr.com/docs/how-to-install-opencv-on-centos-7

https://blog.csdn.net/whudee/article/details/93379780?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-1

# How to Install OpenCV on CentOS 7

Published on: Mon, Oct 23, 2017 at 2:41 pm EST 

CentOSSystem Admin

OpenCV, also known as Open Source Computer Vision Library, is an open source cross-platform computer vision algorithm library. Nowadays, OpenCV is being widely used in all kind of visual processing areas, such as facial recognition, gesture recognition, human-computer interaction, Object identification, motion tracking, etc.

OpenCV can be deployed on various platforms, including Windows, Linux, Android, iOS, etc. In this article, I will show you how to compile and install OpenCV 3.3.0, the latest stable release of OpenCV at the time I wrote this article, on the CentOS 7 x64 operating system.

#### Prerequisites

- A Vultr CentOS 7 x64 server instance.
- Logging in as `root`.
- The server instance has been [updated to the latest stable status](https://www.vultr.com/docs/how-to-update-centos-7-ubuntu-16-04-and-debian-8).

#### Step 1: Install dependencies for OpenCV

Use the following commands to install all required dependencies for compiling OpenCV:

```
yum groupinstall "Development Tools" -y
yum install cmake gcc gtk2-devel numpy pkconfig -y
```

#### Step 2: Download the OpenCV 3.3.0 archive

Download and uncompress OpenCV 3.3.0 archive as below:

```
cd
wget https://github.com/opencv/opencv/archive/3.3.0.zip
unzip 3.3.0.zip
```

#### Step 3: Compile and install OpenCV 3.3.0

Use the following commands to compile and install OpenCV, and compiled OpenCV files will be saved in the `/usr/local` directory.

```
cd opencv-3.3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
make install
```

#### Step 4: Configure required variables

In addtion to compiling and installing files, you need to specify path info for pkgconfig and OpenCV:

```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
echo '/usr/local/lib/' >> /etc/ld.so.conf.d/opencv.conf
ldconfig
```

#### Step 5 (optional): Run tests

To test your OpenCV installation, you can download extra test data from OpenCV extra repository:

```
cd
git clone https://github.com/opencv/opencv_extra.git
export OPENCV_TEST_DATA_PATH=/root/opencv_extra/testdata
```

In the cmake build directory, you will find several test executables named in the same kind of format `opencv_test_*`. Run any one you are interested in to perform a test. For example:

```
cd /root/opencv-3.3.0/build/bin
ls
./opencv_test_photo
```

This concludes the tutorial. Thanks for reading.