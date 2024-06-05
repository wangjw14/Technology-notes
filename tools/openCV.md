# openCV

- 使用conda安装opencv，遇到问题：

  ```
  ImportError: libOpenGL.so.0: cannot open shared object file: No such file or directory
  ```

  解决方法：

  ```sh
  conda install -c fastai opencv-python-headless
  ```

  