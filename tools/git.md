# Git

### 一、git基础

#### git 配置

```shell
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
git config --local user.name "wangjw14"
git config --local user.email "wang.jingwen@outlook.com"
git config --local user.name "wangjingwen03"
git config --local user.email "wangjingwen03@xx.com"
git config --list

git <verb> --help             # 获取帮助
git help <verb>               # 获取帮助
```

#### 创建版本库

```shell
git init
git add <file>
git commit -m <message>
git commit -a -m 'skip add steps'
```

### 二、版本库常用操作

```shell
git status                         # 查看工作区状态
git status -sb -uno --show-stash   # 更友好的输出格式显示
git diff <file>                    # 如果工作区有被修改，可以用该命令查看修改内容
```

#### 版本回退

```shell
git log                           # 查看提交历史
git log --pretty=oneline          # 查看提交历史，一行显示
git reflog                        # 查看所有提交，可以重新回到未来
git reset --hard HEAD^
git reset --hard commit_id        # 回到commit_id的版本
git push origin HEAD --force      # 强制提交现有版本到远端服务器
```

#### 工作区和暂存区

- 工作区（working directory）为一个系统文件夹
- 工作区下的.git目录为版本库（repository），版本库中包含暂存区（stage）和分支（默认为master）等
- ```git add```命令用于将修改放到暂存区，然后执行```git commit```可以将暂存区的修改提交到分支
- ```git diff HEAD -- file```命令可以用于查看工作区和版本库里最新版本的区别

#### 撤销修改

```shell
git checkout -- file        # 用于丢弃工作区的修改（实质是，用版本库里的版本替换工作区的版本）
git reset HEAD <file>       # 用于丢弃暂存区的修改，回到了可以使用上一行代码的情况
```

#### 删除文件

```sh
git rm         # 用于删除文件
```

### 三、远程仓库

```shell
ssh-keygen -t rsa -C "email@example.com"    # 生成id_rsa和id_raa.pub文件
ssh-keygen -t rsa -C 'wangjingwen03@xx.com' -f ~/.ssh/id_rsa.wangjingwen03
# 在github或者gitlab上将公钥添加到账户，并创建仓库
git remote add origin git@github.com:michaelliao/learn.git    # 将本地库和远程库进行关联
git push -u origin master          # 第一次推送master到远程，-u参数关联本地分支和远程分支
git push origin master             # 之后推送master到远程
git push <remote hostname> <local branch>:<remote branch>
git clone url                      # 从远程克隆一个仓库到本地
                                # 使用ssh，而不是https，可以避免每次都输入密码
```

### 四、分支管理

- 开发一个新feature，最好新建一个分支。

```shell
git branch                        # 查看所有分支
git branch -a -v                  # 查看本地和远程所有分支，并显示最后一次提交信息
git branch <name>                 # 创建分支
git checkout <name>               # 切换分支
git checkout -b dev               # 创建+切换分支
git merge <name>                  # 合并某分支到当前分支
git branch -d <name>              # 删除分支

git log --oneline --graph         # 查看日志
git log --graph --pretty=oneline --abbrev-commit   # 查看分支的合并情况

git merge --no-ff -m "merge with no-ff" <name>     # 使用no ff方式进行合并，保留分支信息

git stash                         # 保存工作现场，之后可以用于常见bug分支，用于修复bug
git stash pop                     # 恢复工作现场
git stash list                    # 查看保存了哪些现场
git branch -D <name>              # 强行删除未合并的分支，如feature分支
```

#### 多人协作

```shell
git remote                        # 查看远程库的信息
git remote -v                     # 查看远程库详细信息
git remote rm origin              # 删除已有的远程库origin
git remote update origin          # 本地拉取远程的更新
git branch -a                     # 查看远程库的分支信息
git push origin branch-name       # 推送本地信息到远端分支
git pull                          # 上条推送失败，则需要先抓取远程的新提交
git checkout -b name origin/name  # 在本地创建和远程分支对应的分支
git branch --set-upstream name origin/name # 建立本地和远端分支的关联
git rebase                        # 将本地未push的分叉提交历史整理成直线
```

### 五、标签管理

```shell
git tag <tagname>                 # 用于创建一个新的标签，默认为HEAD，也可以指定一个commit id
git tag -a <tagname> -m "blabla"  # 创建标签，并指定相应的信息
git tag                           # 查看使用标签

git push origin <tagname>         # 推送一个本地标签
git push origin --tags            # 推送所有本地未推送的标签
git tag -d <tagname>              # 删除一个本地标签
git push origin :refs/tags/<tagname>  # 删除一个远程标签
```

### 六、自定义git

```shell
git add -f <filename>            # 强制将.gitignore文件中的文件加入暂存区
git check-ignore -v <filename>   # 检查.gitignore文件中哪个规则忽略了文件

git config --global alias.co checkout           # 为git命令添加别名
git config --global alias.cm commit             # 每个仓库的git配置文件位置：.git/config
git config --global alias.unstage 'reset HAED'  # 当前用户的git配置文件位置：~/.gitconfig

git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

### git技巧

- git clone加速
  
  ```sh
  # socks5协议，1080端口修改成自己的本地代理端口（使用shadowsocks时，使用下面的命令）
  git config --global http.https://github.com.proxy socks5://127.0.0.1:7891
  git config --global https.https://github.com.proxy socks5://127.0.0.1:7891
  
  # http协议，1081端口修改成自己的本地代理端口
  git config --global http.https://github.com.proxy https://127.0.0.1:7890
  git config --global https.https://github.com.proxy https://127.0.0.1:7890
  
  # 使用https的网址，进行git clone。
  git clone https://github.com/wangjw14/Summary-notes.git
  ```

- 同一客户端下使用多个git账号
  
  - https://www.jianshu.com/p/89cb26e5c3e8

- 拉取远程指定分支
  
  - https://zj-git-guide.readthedocs.io/zh_CN/latest/advanced/%E6%8B%89%E5%8F%96%E6%8C%87%E5%AE%9A%E8%BF%9C%E7%A8%8B%E5%88%86%E6%94%AF%E5%88%B0%E6%9C%AC%E5%9C%B0/
  
  - 新建仓库
    
    ```shell
    $ mkdir gitrepo
    $ cd giterpo
    $ git init
    ```
    
    拉取远程指定分支
    
    ```sh
    $ git remote add origin https://github.com/zjZSTU/zjzstu.github.com.git
    $ git remote add origin ssh://wangjingxxx@icode.XXXX/xxx/feed-cv/semantic
    $ # 执行安装hook代码
    ¥ git fetch origin dev
    ```
    
    新建本地分支并关联到指定远程分支
    
    ```shell
    $ git checkout -b dev origin/dev
    ```

- Hi wangjw14! You've successfully authenticated, but GitHub does not provide shell access.

  ```sh
  git remote set-url origin git@github.com:wangjw14/Technology-notes.git
  # https://stackoverflow.com/questions/26953071/github-authentication-failed-github-does-not-provide-shell-access
  ```

- git 解决中文乱码

  ```shell
  git config --global core.quotepath false
  ```

  