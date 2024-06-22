# Linux Command

- [Shell 教程 | 菜鸟教程](https://www.runoob.com/linux/linux-shell.html)

## 设置新的home和目录

```shell
# 原先home下的.bashrc
alias wangjw='. /ssd3/work/wangjingwen03/.bashrc'


# 新目录下的.bashrc
export HOME=/ssd3/work/wangjingwen03

alias python3="/ssd3/work/wangjingwen03/python_env_3/bin/python"
alias pip3="/ssd3/work/wangjingwen03/python_env_3/bin/pip"


#hadoop
alias hadoopyinglong="hadoop fs -D hadoop.job.ugi=xx,xx -D fs.default.name=afs://yinglong.afs.baidu.com:9902"
alias hadoopocr="hadoop fs -conf /home/work/wangjingwen03/git_repo/wangjingwen03/short_video_segment/shuaku/hadoop_script_ocr_get/hadoop-site.xml"
alias hadoopbaihua="hadoop fs -conf /home/work/wangjingwen03/git_repo/wangjingwen03/short_video_segment/shuaku/hadoop_script_page_B/hadoop-site.xml"
```

## 查看系统版本

```sh
cat /etc/os-release
lsb_release -a
cat /proc/version
```



## awk

```shell
awk '$24 == 1 {print $1}' part-39960 > part-39960-vip
```

```shell
# 文件去重
awk '!a[$0]++' file1.txt 
```

https://www.ruanyifeng.com/blog/2018/11/awk.html

```shell
# 求和
cat data|awk '{sum+=$1} END {print "Sum = ", sum}'
```

https://blog.csdn.net/csCrazybing/article/details/52594989

awk

- [awk 入门教程 - 阮一峰的网络日志](https://www.ruanyifeng.com/blog/2018/11/awk.html)

## sed

```shell
sed -n '1501,2000p' pandian_labeling.txt > pandian_labeling_500_4.txt
```

```shell
# 删除第N行：
sed -i 'Nd' filename 

# 删除第M到N行：
sed -i 'M,Nd' filename

# 删除a.txt中含"abc"的行，但不改变a.txt文件本身，操作之后的结果在终端显示
sed -e '/abc/d'  a.txt   

# 删除a.txt中含"abc"的行，将操作之后的结果保存到a.log
sed -e '/abc/d'  a.txt  > a.log   

# 删除含字符串"abc"或“efg"的行，将结果保存到a.log
sed '/abc/d;/efg/d' a.txt > a.log    
```

## date

```shell
TODAY=$(date "+%Y%m%d")
LOAD_DATE=$(date -d"2 day ago ${TODAY}" +%Y%m%d)
```

## sort

```shell
sort [-r] temp/file1.txt -o temp/file1.txt
```



```sh
# 一、交集
sort a.txt b.txt | uniq -d

# 二、并集
sort a.txt b.txt | uniq 

# 三、差集
# a.txt-b.txt:
sort a.txt b.txt b.txt | uniq -u
# b.txt - a.txt:
sort b.txt a.txt a.txt | uniq -u
```

四、相关的解释
使用sort可以将文件进行排序，可以使用sort后面的玲玲，例如 -n 按照数字格式排序，例如 -i 忽略大小写，例如使用-r 为逆序输出等

uniq为删除文件中重复的行，得到文件中唯一的行，后面的命令 -d 表示的是输出出现次数大于1的内容 -u表示的是输出出现次数为1的内容，那么对于上述的求交集并集差集的命令做如下的解释：

sort a.txt b.txt | uniq -d：将a.txt b.txt文件进行排序，uniq使得两个文件中的内容为唯一的，使用-d输出两个文件中次数大于1的内容，即是得到交集

sort a.txt b.txt | uniq ：将a.txt b.txt文件进行排序，uniq使得两个文件中的内容为唯一的，即可得到两个文件的并集

sort a.txt b.txt b.txt | uniq -u：将两个文件排序，最后输出a.txt b.txt b.txt文件中只出现过一次的内容，因为有两个b.txt所以只会输出只在a.txt出现过一次的内容，即是a.txt-b.txt差集

对于b.txt-a.txt为同理

## uniq

```sh
sort test.txt | uniq
```

## comm

```sh
comm [-123] file1 file2

# comm -1 不显示只出现在第一个文件的行。
# comm -2 不显示只出现在第二个文件的行。
# comm -3 不显示同时出现在两个文件的行。
# comm file1 file2 显示三列，第一列代表只出现在file1的行，第二列代表只出现在file2的行，第三列代表俩个文件同时出现的行
# comm -12 显示两个文件同时出现的行 也就是交集
# comm -13 显示只出现在第二个文件的行
# comm -23 显示只出现在第一个文件的行
```

| Command | Description                |
| ------- | -------------------------- |
| du -sh  | 查看文件夹的大小，s表示summrize，仅显示总计 |
| df -h   | 查看磁盘空间的使用情况                |
| free -h | 查看内存大小                     |
| top     | 查看内存使用情况                   |

## tail

```shell
tail -f filename
tail -n +100 filename
```

## split

```shell
split -l 1000000 file_name -d -a 3 file_name_part_
```

## 在mac上挂在ntfs格式的磁盘

```shell
sudo umount /Volumes/UNTITLED
sudo mount -t ntfs -o rw,auto,nobrowse /dev/disk3s1 ~/ntfs-volume
sudo umount ~/ntfs-volume
```

## 压缩相关命令

- tar
  
  | Command | Description                                                                                                                                                                                               |
  | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | tar     | `-z(gzip)`      用gzip来压缩/解压缩文件<br/>`-j(bzip2)`     用bzip2来压缩/解压缩文件<br/>`-v(verbose)`   详细报告tar处理的文件信息<br/>`-c(create)`    创建新的档案文件<br/>`-x(extract)`   解压缩文件或目录<br/>`-f(file)`      使用档案文件或设备，这个选项通常是必选的。 |
  
  举例
  
  ```shell
  #压缩
  [root@localhost tmp]# tar -zvcf buodo.tar.gz buodo
  [root@localhost tmp]# tar -jvcf buodo.tar.bz2 buodo 
  
  #解压
  [root@localhost tmp]# tar -zvxf buodo.tar.gz 
  [root@localhost tmp]# tar -jvxf buodo.tar.bz2
  [root@localhost tmp]# tar –xvf buodo.tar
  ```

- gzip 
  
  | Command | Description                             |
  | ------- | --------------------------------------- |
  | gzip    | 压缩后的格式为：*.gz <br/>这种压缩方式不能保存原文件；且不能压缩目录 |
  
  举例
  
  ```shell
  #压缩
  [root@localhost tmp]# gzip buodo
  [root@localhost tmp]# ls
  buodo.gz
  #解压
  [root@localhost tmp]# gunzip buodo.gz 
  [root@localhost tmp]# ls
  buodo
  ```

- zip
  
  | Command | Description                                                          |
  | ------- | -------------------------------------------------------------------- |
  | zip     | 与gzip相比：1）可以压缩目录； 2）可以保留原文件； <br/>`-r(recursive)`    递归压缩目录内的所有文件和目录 |
  
  举例
  
  ```shell
  #压缩和解压文件
  [root@localhost tmp]# zip boduo.zip boduo
  [root@localhost tmp]# unzip boduo.zip
  
  #压缩和解压目录
  [root@localhost tmp]# zip -r Demo.zip Demo
    adding: Demo/ (stored 0%)
    adding: Demo/Test2/ (stored 0%)
    adding: Demo/Test1/ (stored 0%)
    adding: Demo/Test1/test4 (stored 0%)
    adding: Demo/test3 (stored 0%)
  [root@localhost tmp]# unzip Demo.zip 
  Archive:  Demo.zip
     creating: Demo/
     creating: Demo/Test2/
     creating: Demo/Test1/
   extracting: Demo/Test1/test4        
   extracting: Demo/test3  
  ```

- bzip2
  
  | Command | Description                            |
  | ------- | -------------------------------------- |
  | bzip2   | 压缩后的格式：.bz2； <br/>`-k`    产生压缩文件后保留原文件 |
  
  举例
  
  ```shell
  #压缩
  [root@localhost tmp]# bzip2 boduo
  [root@localhost tmp]# bzip2 -k boduo
  
  #解压
  [root@localhost tmp]# bunzip2 boduo.bz2 
  ```

- 参考资料：https://blog.csdn.net/capecape/article/details/78548723 

- 删除大小为0的文件
  
  ```shell
  find . -name "*" -type f -size 0c | xargs -n 1 rm -f
  ```
  
  参考资料：https://www.cnblogs.com/webStyle/p/4361982.html

- grep
  
  ```
  grep -oP "video_cutframe_cost.*? "
  ```
  
  ## 阮一峰的linux命令介绍

https://www.bookstack.cn/read/bash-tutorial/docs-archives-command.md

## json处理

https://hellogitlab.com/OS/Centos/json_tool_jq.html#_1-%E5%AE%89%E8%A3%85

jq

## 目录相关

```shell
cur_dir=$(dirname $(readlink -f "$0"))   # 当前文件的路径
`pwd`
```

## corntab

## 查看系统版本

```shell
lsb_release -a
uname -a
```

## 设置语言编码

准备工作1：安装操作系统后设置系统的默认编码LANG="en_US.UTF-8"

```shell
vi /etc/sysconfig/i18n
LANG="en_US.UTF-8"
```

准备工作2：修改vim的默认配置~/.vimrc，避免每次需要set enc=utf8

```shell
vim ~/.vimrc
set encoding=utf-8
```

## 下载各种包

rpm包下载地址，注意下载x86.64 centos版本的

http://www.rpmfind.net/linux/rpm2html/search.php?query=openssl&submit=Search+...&system=&arch=

```shell
#安装glibc相关的库

glibc-2.17-260.el7.x86_64.rpm
glibc-common-2.17-260.el7.x86_64.rpm
glibc-devel-2.17-260.el7.x86_64.rpm
glibc-headers-2.17-260.el7.x86_64.rpm
libstdc++-4.8.5-36.el7.x86_64.rpm

rpm -Uvh --force --nodeps glibc*.rpm
rpm -Uvh --force --nodeps libstdc++-4.8.5-36.el7.x86_64.rpm

# 这个可以选择安装
rpm -Uvh --force --nodeps nscd-2.17-260.el7.x86_64.rpm

#安装openssl，不然wget会有问题，python3也有依赖
openssl-1.0.1e-57.el6.x86_64.rpm
openssl-devel-1.0.1e-57.el6.x86_64.rpm

rpm -Uvh --force --nodeps openssl*.rpm
```

## 查找文件

```shell
find / -name tk8.5
usr/lib64/tk8.5

find / -name tcl8.5
/usr/share/tcl8.5

find . -name "*.c"

```

## 配置jupyter notebook远程可访问（如果需要）

```shell
#初始化jupyter配置文件
jupyter notebook --generate-config
#生成密码
jupyter notebook password

 # 注意目录根据自己实际确定
$vim ~/.jupyter/jupyter_notebook_config.py

c.NotebookApp.ip='你机器的监控ip'
c.NotebookApp.ip='*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #随便指定一个端口

# 开启代码补全
# 1.  安装显示目录功能 pip install jupyter_contrib_nbextensions 
# 2. 安装 nbextention     jupyter contrib nbextension install --user --skip-running-check 
# 3. 启动 Jupyter Notebook，上面选项栏会出现 Nbextensions 的选项，并勾选 Hinterland
```

## du

```shell
du -sh
du -sh *
du -sh * | sort -nr
du -h --max-depth=1
du -h --max-depth=1 | grep [TG] |sort
```

## free

```sh
# 检查机器的内存
free -h
```

## df 

```sh
# 查看磁盘空间
df -h
```

## netstat

```sh
# 查看端口信息
netstat -ntlp
```

## 批量杀死进程

```sh
ps -ef | grep ld-linux-x86-64 | awk '{print $2}' | head -2 | xargs  kill -9
```

## ffmpeg

```shell
ffmpeg -ss [start_time] -to [end_time] -i [input_file] -c copy [output_file]
ffmpeg -ss [start_time] -t [duration] -i [input_file] -c copy [output_file]
```

## 批量启动任务

```sh
for i in $(seq 0 7)
do
    echo $i
    sed -i "12s/gpu: 0;1/gpu: ${i};1/" conf/models.conf
    sed -i "4s/port: 8700/port: 870$i/" conf/service.conf
    sh bin/control.sh restart --platform=gpu_a10
    sed -i "12s/gpu: ${i};1/gpu: 0;1/" conf/models.conf
    sed -i "4s/port: 870$i/port: 8700/" conf/service.conf
done

# 批量调用

for i in $(seq 0 7)
do 
    python3 req_summary_V1.py $i &
done

wait

```

## tmux

- https://www.ruanyifeng.com/blog/2019/10/tmux.html

```shell
新建会话tmux new -s my_session。
在 Tmux 窗口运行所需的程序。
按下快捷键Ctrl+b d将会话分离。
下次使用时，重新连接到会话tmux attach-session -t my_session。
```





