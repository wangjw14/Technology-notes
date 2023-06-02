# python环境

前往 [Python源码下载页面](https://www.python.org/downloads/source/) 下载源码

```sh
tar -xzvf Python-3.8.6.tgz
cd Python-3.8.6
./configure  --prefix=/home/yuanhao03/Python3.8.6 --enable-shared LDFLAGS=-Wl,-rpath=/home/yuanhao03/Python3.8.6/lib --with-system-ffi
make 
# make clean  如果编译遇到一些问题，或者某一些模块没有编译进去，重新编译之前最好clean一下
make install
```

--enable-shared相关文档 https://www.yuque.com/shenweiyan/cookbook/python-enable-shared

安装完成后会提示pip和easy_install在/home/yuanhao03/Python3.8.6/bin目录下，这个时候需要我们把『export PATH="/home/yuanhao03/Python3.8.6/bin:$PATH』添加到.bashrc中，并source生效。

然后，对Python3.8.6目录打包"tar -zcvf python3-8-6.tar.gz Python3.8.6"，上传到afs路径下即可在hadoop提交任务中配置使用。

同上，在~目录下创建mkdir .pip目录，然后创建pip.conf文件，并把公司pip源配置