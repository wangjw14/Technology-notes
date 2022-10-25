# FTP

ftp

- 创建一个samba的账户（也可以用其他名字，后续配置响应修改即可）：
  
  ```shell
  $ useradd samba                 # 添加samba账户
  $ passwd samba                  # 修改samba密码
  $ chsh samba -s /sbin/nologin   # 设置账户不可登录
  $ usermod -d / samba            # 设置账户的根目录为/， 这个很重要，ftp链接中的uri跟路径是账户的home路径
  ```

- 修改/home/samba的目录权限:
  
  ```shell
  chmod -R 777 /home/samba/
  ```

- 搭建ftp服务器有两种方案，这里使用prodtpd。以root用户登录开发机，并使用yum安装proftpd。
  
  ```shell
  yum install proftpd #安装proftpd并启动proftpd服务
  ```

- 为了支持匿名访问下载，需要修改配置，根据参考，修改/etc/proftpd.conf配置如下：
  
  ```
  ServerName                          "ProFTPD"
  ServerType                          standalone
  DefaultServer                       on
  
  timesGMT off
  # Port 21 is the standard FTP port.
  Port                                21
  # Umask 022 is a good standard umask to prevent new dirs and files
  # from being group and world writable.
  Umask                               022
  IdentLookups                        off
  UseReverseDNS                       off
  # To prevent DoS attacks, set the maximum number of child processes
  # to 30.  If you need to allow more than 30 concurrent connections
  # at once, simply increase this value.  Note that this ONLY works
  # in standalone mode, in inetd mode you should use an inetd server
  # that allows you to limit maximum number of processes per service
  # (such as xinetd)
  MaxInstances                        30
  
  # Set the user and group that the server normally runs at.
  User                                samba
  Group                               samba
  ```

- Normally, we want files to be overwriteable.

<Directory /*> AllowOverwrite on

- A basic anonymous configuration, no upload directories.

<Anonymous ~samba>
 User samba
 Group samba

- We want clients to be able to login with "anonymous" as well as "ftp"

UserAlias anonymous samba

```
  # Limit the maximum number of anonymous logins
  MaxClients                        30

  # We want 'welcome.msg' displayed at login, and '.message' displayed
  # in each newly chdired directory.
  DisplayLogin                      welcome.msg

  # Limit WRITE everywhere in the anonymous chroot
  <Limit WRITE>
      DenyAll
  </Limit>
  <Directory /etc>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /lib64>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /lib>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /bin>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /sbin>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /usr>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /var>
      <Limit ALL>
          DenyAll
      </Limit>
  </Directory>

  <Directory /home>
      <Limit LIST>
          AllowAll
      </Limit>
  </Directory>

  <Directory /flash/>
      <Limit LIST>
          DenyAll
      </Limit>
  </Directory>

  <Directory />
      <Limit LIST>
          DenyAll
      </Limit>
  </Directory>

  <Directory /opt>
      <Limit LIST>
          DenyAll
      </Limit>
  </Directory>

  <Limit WRITE>
      DenyAll
  </Limit>
```

```
- 根据第92行配置只开启了/home/samba下的共享，因此需要将共享文件放在/home/samba下。

- 启动

  ```shell
  service proftpd start
```
