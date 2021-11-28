#Centos升级cmake

今天在编译 Opencv 时，在 cmake 阶段失败，报类似下面的错误信息：

CMake 3.5.2 or higher is required. You are running version 3.4.0

很明显，这是 cmake 版本过低导致的，那么就需要升级 cmake 到更高的版本。下面是升级 cmake 的步骤：

1.卸载原有通过 yum 安装的 cmake

yum remove cmake

2.下载cmake安装包（直接到网址下载）

wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5-Linux-x86_64.tar.gz

3.解压下载好的cmake二进制包（本例使用的是cmake3.14.5版本，软件包路径： /opt ）

cd /opt

tar zxvf cmake-3.14.5-Linux-x86_64.tar.gz

4.解压成功后，就可以在 /opt 目录下看到一个 cmake-3.14.5-Linux-x86_64 目录，下面添加cmake环境变量，编辑 /etc/profile.d/cmake.sh 文件，写入以下内容:

```shell
export CMAKE_HOME=/opt/cmake-3.14.5-Linux-x86_64
export PATH=$PATH:$CMAKE_HOME/bin
```

5.保存并退出，执行命令让 cmake 环境文件生效

source /etc/profile

6.此时，再次查看cmake版本，就已经是 3.14.5 了：

cmake -version

至此cmake 版本升级完毕。





# [cmake的一个编译报错](https://www.cnblogs.com/minglee/p/9016306.html)

在一台新搭建的服务器上执行cmake的时候，报了如下错误：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
$ cmake ./
-- The C compiler identification is unknown
-- The CXX compiler identification is GNU 4.4.7
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- broken
CMake Error at /usr/share/cmake/Modules/CMakeTestCCompiler.cmake:61 (message):
The C compiler "/usr/bin/cc" is not able to compile a simple test program.

It fails with the following output:

...
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

查看下gcc与g++的版本：

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
$ gcc --version
gcc (GCC) 5.1.0
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

$ g++ --version
g++ (GCC) 5.1.0
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

发现都是5.1.0，那为何会有这行“The CXX compiler identification is GNU 4.4.7”报错呢？

查看当前目录下的CMakeCache.txt

发现如下两行配置：

```
//CXX compiler.
CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++

//C compiler.
CMAKE_C_COMPILER:FILEPATH=/usr/bin/cc
```

执行 /usr/bin/c++ --version 和 /usr/bin/cc --version，发现输出的版本号仍然是5.1.0，这就有点莫名其妙了。

google搜索出了一个github issue：https://github.com/Kingsford-Group/genesum/issues/2，在里面找到了解决方案：

```
cmake -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc) ..
```

执行之后果然可以了，并且重新打开了CMakeCache.txt之后发现，编译器的两个选项改变了：

```
//CXX compiler.
CMAKE_CXX_COMPILER:FILEPATH=/usr/local/bin/g++

//C compiler.
CMAKE_C_COMPILER:FILEPATH=/usr/local/bin/gcc
```

这两个路径与命令 which gcc 和 which g++的输出一致。

猜测手动改CMakeCache.txt 的这两项应该也可以解决问题，比较困惑的就是，为何运行/usr/bin/c++ --version得到的版本号仍然是5.1.0？

这个疑惑要留待以后来解决了。