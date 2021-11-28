# C++ code

- 传给函数一个指针，并要在函数内部指向一个创建数组，函数退出时，要将数组的值复制给一段指针所指的空间。否则数组会被析构。

  ```c++
  hi_img->rgb_data = (unsigned char*)malloc(cv_img_roi.rows * cv_img_roi.cols * 3);
  memcpy(hi_img->rgb_data, cv_img_roi.data, cv_img_roi.rows * cv_img_roi.cols * 3);
  ```

  

- 输出指针所指的一段空间的内容

  ```c++
  unsigned char* f1 = (unsigned char *) (hi_img1.rgb_data) ;
  for (int i=0;i<20;i++){
      std::cout<< int(f1[i]) << " ";
  }
  std::cout<< std::endl;
  ```



- 多线程

  使用thread无返回值版本

  ```c++
  #include <iostream>
  #include <numeric>
  #include <thread>
  #include <vector>
  
  #include "src/lib/utility.h"
  
  // A demo for creating two threads
  // Run this using one of the following methods:
  //  1. With bazel: bazel run src/main:vector_of_threads_main
  //  2. With plain g++: g++ -std=c++17 -lpthread src/main/vector_of_threads_main.cc  -I ./
  int main() {
    const int number_of_threads = 1000;
    uint64_t number_of_elements = 1000 * 1000* 1000;
    uint64_t step = number_of_elements / number_of_threads;
    std::vector<std::thread> threads;
    std::vector<uint64_t> partial_sums(number_of_threads);
  
    for (uint64_t i = 0; i < number_of_threads; i++) {
      threads.push_back(std::thread(AccumulateRange, std::ref(partial_sums[i]),
                                    i * step, (i + 1) * step));
    }
  
    for (std::thread &t : threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  
    uint64_t total =
        std::accumulate(partial_sums.begin(), partial_sums.end(), uint64_t(0));
  
    PrintVector(partial_sums);
    std::cout << "total: " << total << std::endl;
  
    return 0;
  }
  ```

  使用async有返回值版本

  ```c++
  #include <iostream>
  #include <numeric>
  #include <thread>
  #include <vector>
  #include <future>
  
  #include "src/lib/utility.h"
  
  // A demo for creating two threads
  // Run this using one of the following methods:
  //  1. With bazel: bazel run src/main/mutex:{THIS_FILE_NAME_WITHOUT_EXTENSION}
  //  2. With plain g++: g++ -std=c++17 -lpthread
  //  src/main/mutex/{THIS_FILE_NAME}.cc  -I ./
  int main() {
    const int number_of_threads = 20;
    uint64_t number_of_elements = 1000 * 1000 * 1000;
    uint64_t step = number_of_elements / number_of_threads;
    std::vector<std::future<uint64_t>> tasks;
  
    for (uint64_t i = 0; i < number_of_threads; i++) {
      tasks.push_back(std::async(GetRangeSum, i * step, (i + 1) * step));
    }
    
    uint64_t total = 0;
    for (auto &t : tasks) {
      auto p = t.get();
      std::cout << "p: " << p << std::endl;
      total += p;
    }
  
    std::cout << "total: " << total << std::endl;
  
    return 0;
  }
  ```

  参考资料：

  https://github.com/ourarash/multithreading_cpp/tree/master/src/main 

