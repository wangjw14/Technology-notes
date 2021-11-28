# Docker基础命令

- ### image 文件

  **Docker 把应用程序及其依赖，打包在 image 文件里面。**

  ```shell
  # 列出本机的所有 image 文件。
  $ docker image ls
  $ docker iamges
  
  # 删除 image 文件
  $ docker image rm [imageName]
  ```

- ### 容器文件

  **image 文件生成的容器实例，本身也是一个文件，称为容器文件。**

  ```shell
  # 列出本机正在运行的容器
  $ docker container ls
  
  # 列出本机所有容器，包括终止运行的容器
  $ docker container ls --all
  
  # 删除终止运行的容器文件
  $ docker container rm [containerID]
  ```




- ### 制作自己的docker容器 

  准备工作，下载源码。

  ```shell
  $ git clone https://github.com/ruanyf/koa-demos.git
  $ cd koa-demos
  ```

  1. 编写Dockerfile文件

     首先，在项目的根目录下新建一个`.dockerignore` 文件，写入不包含如镜像的内容。

     ```
     .git
     node_modules
     npm-debug.log
     ```

     然后，新建一个文本文件Dockerfile

     ```dockerfile
     FROM node:8.4
     COPY . /app
     WORKDIR /app
     RUN npm install --registry=https://registry.npm.taobao.org
     EXPOSE 3000
     ```

     上面代码一共五行，含义如下。

     > - `FROM node:8.4`：该 image 文件继承官方的 node image，冒号表示标签，这里标签是`8.4`，即8.4版本的 node。
     > - `COPY . /app`：将当前目录下的所有文件（除了`.dockerignore`排除的路径），都拷贝进入 image 文件的`/app`目录。
     > - `WORKDIR /app`：指定接下来的工作路径为`/app`。
     > - `RUN npm install`：在`/app`目录下，运行`npm install`命令安装依赖。注意，安装后所有的依赖，都将打包进入 image 文件。
     > - `EXPOSE 3000`：将容器 3000 端口暴露出来， 允许外部连接这个端口。

  2. 创建image文件

     ```shell
     $ docker image build -t koa-demo .
     # 或者
     $ docker image build -t koa-demo:0.0.1 .
     ```

     `-t`参数用来指定 image 文件的名字，后面还可以用冒号指定标签。如果不指定，默认的标签就是`latest`。最后的那个点表示 Dockerfile 文件所在的路径，上例是当前路径，所以是一个点。

  3. 生成容器

     ```shell
     $ docker container run -p 8000:3000 -it koa-demo /bin/bash
     # 或者
     $ docker container run -p 8000:3000 -it koa-demo:0.0.1 /bin/bash
     ```

     上面命令的各个参数含义如下：

     > - `-p`参数：容器的 3000 端口映射到本机的 8000 端口。
     > - `-it`参数：容器的 Shell 映射到当前的 Shell，然后你在本机窗口输入的命令，就会传入容器。
     > - `koa-demo:0.0.1`：image 文件的名字（如果有标签，还需要提供标签，默认是 latest 标签）。
     > - `/bin/bash`：容器启动以后，内部第一个执行的命令。这里是启动 Bash，保证用户可以使用 Shell。

     如果一切正常，运行上面的命令以后，就会返回一个命令行提示符。

      ```bash
      root@66d80f4aaf1e:/app#
      ```
     
  4. 退出和删除容器

     按下 Ctrl + d （或者输入 exit）退出容器。此外，也可以用`docker container kill`终止容器运行。

     ```shell
     # 在本机的另一个终端窗口，查出容器的 ID
     $ docker container ls
     # 停止指定的容器运行
     $ docker container kill [containerID]
     ```

     容器停止运行之后，并不会消失，用下面的命令删除容器文件。
     
     ```shell
     # 查出容器的 ID
     $ docker container ls --all
     
     # 删除指定的容器文件
     $ docker container rm [containerID]
     ```
     
     也可以使用`docker container run`命令的`--rm`参数，在容器终止运行后自动删除容器文件。
     
     ````shell
     $ docker container run --rm -p 8000:3000 -it koa-demo /bin/bash
     ````
     
  5.  CMD命令
  
     容器启动之后，可以自行执行一个命令，将其写入Dockerfile文件里面
  
      ```bash
      FROM node:8.4
      COPY . /app
      WORKDIR /app
      RUN npm install --registry=https://registry.npm.taobao.org
      EXPOSE 3000
      CMD node demos/01.js
      ```
  
     上面的 Dockerfile 里面，多了最后一行`CMD node demos/01.js`，它表示容器启动后自动执行`node demos/01.js`。
  
     你可能会问，`RUN`命令与`CMD`命令的区别在哪里？简单说，`RUN`命令在 image 文件的构建阶段执行，执行结果都会打包进入 image 文件；`CMD`命令则是在容器启动后执行。另外，一个 Dockerfile 可以包含多个`RUN`命令，但是只能有一个`CMD`命令。
  
     注意，指定了`CMD`命令以后，`docker container run`命令就不能附加命令了（比如前面的`/bin/bash`），否则它会覆盖`CMD`命令。现在，启动容器可以使用下面的命令。
  
      ```bash
      $ docker container run --rm -p 8000:3000 -it koa-demo:0.0.1
      ```
  
  
  
- ### 其他的有用命令

  **（1）docker container start**

  前面的`docker container run`命令是新建容器，每运行一次，就会新建一个容器。同样的命令运行两次，就会生成两个一模一样的容器文件。如果希望重复使用容器，就要使用`docker container start`命令，它用来启动已经生成、已经停止运行的容器文件。

   ```bash
   $ docker container start [containerID]
   ```

  **（2）docker container stop**

  前面的`docker container kill`命令终止容器运行，相当于向容器里面的主进程发出 SIGKILL 信号。而`docker container stop`命令也是用来终止容器运行，相当于向容器里面的主进程发出 SIGTERM 信号，然后过一段时间再发出 SIGKILL 信号。

   ```bash
   $ docker container stop [containerID]
   ```

  这两个信号的差别是，应用程序收到 SIGTERM 信号以后，可以自行进行收尾清理工作，但也可以不理会这个信号。如果收到 SIGKILL 信号，就会强行立即终止，那些正在进行中的操作会全部丢失。

  **（3）docker container logs**

  `docker container logs`命令用来查看 docker 容器的输出，即容器里面 Shell 的标准输出。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令查看输出。

   ```bash
   $ docker container logs [containerID]
   ```

  **（4）docker container exec**

  `docker container exec`命令用于进入一个正在运行的 docker 容器。如果`docker run`命令运行容器的时候，没有使用`-it`参数，就要用这个命令进入容器。一旦进入了容器，就可以在容器的 Shell 执行命令了。

   ```bash
   $ docker container exec -it [containerID] /bin/bash
   ```

  **（5）docker container cp**

  `docker container cp`命令用于从正在运行的 Docker 容器里面，将文件拷贝到本机。下面是拷贝到当前目录的写法。

   ```bash
   $ docker container cp [containID]:[/path/to/file] .
   ```




- 将一个机器上的container移植到另一个机器上

  1. Export the container to a tarball

     ```shell
     $ docker export <CONTAINER ID> > /home/export.tar
     ```

  2. Move your tarball to new machine

  3. Import it back

     ```shell
     $ cat /home/export.tar | docker import - some-name:latest
     ```

     You cannot move a running docker container from one host to another.

     You can commit the changes in your container to an image with [`docker commit`](https://docs.docker.com/engine/reference/commandline/commit/), move the image onto a new host, and then start a new container with [`docker run`](https://docs.docker.com/engine/reference/commandline/run/). This will preserve any data that your application has created inside the container.

     **Nb:** It does not preserve data that is stored inside volumes; you need to move data volumes manually to new host.

- 导出和导入一个docker image

  1. 导出

     ```shell
     docker save -o rocketmq.tar rocketmq 
     ##-o：指定保存的镜像的名字；rocketmq.tar：保存到本地的镜像名称；rocketmq：镜像名字，通过"docker images"查看
     ```

  2. 导入

     ```sh
     docker load --input rocketmq.tar 
     # 或者
     docker load < rocketmq.tar
     ```

  3. 删除

     ```sh
     docker rmi -f image_id ##-f：表示强制删除镜像；image_id：镜像id
     ```

  4. 参考资料：

     Docker 本地导入镜像/保存镜像/载入镜像/删除镜像 https://www.cnblogs.com/linjiqin/p/8604756.html


- ### 一个docker部署例子

  - 用gunzip命令或者解压软件解压缩，得到扩展名为.tar的文件。

    ```shell
    $ gunzip tel_ocr_docker_ubuntu18_v1.2-20200420.tar.gz
    ```

  - 然后用docker命令导入这个容器到宿主机的docker中。

    ```shell
    $ docker import tel_ocr_docker_ubuntu18_v1.2-20200420.tar tel_ocr:v1.2
    ```

  - 启动这个docker容器并运行服务，根据需要可绑定docker服务端口6040到宿主机空闲端口(如8040)。

    ```sh
    $ docker run -itd --name myocr --network bridge -p 8040:6040  tel_ocr:v1.2  bash -c "cd /opt/tel_ocr&&./run.sh"
    ```

  - 检查服务是否正常启动

    ```shell
    $ docker exec -it myocr /bin/bash
    $ cd /opt/tel_ocr
    $ python sigle_file_test.py
    ```

    如果输出ocr的结果则服务正常运行。



- 启动容器，并启动一个bash交互终端

  ```sh
  docker run -it  tel_ocr:v1.1 /bin/bash
  # docker run -it  tel_ocr:v1.1 /bin/bash
  ```

  

### 参考资料 

[Docker 入门教程（阮一峰）](https://www.ruanyifeng.com/blog/2018/02/docker-tutorial.html) 

