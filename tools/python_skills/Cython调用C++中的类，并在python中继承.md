# Cython调用C++中的类，并在python中继承

- InferenceFrozenGraph.h

```c++
#ifndef INFERENCEFROZENGRAPH_H
#define INFERENCEFROZENGRAPH_H
#include <vector>
#include <string>
#include "include/tf_easy.h"

namespace  inference{

    class InferenceFrozenGraph_c{
    public:
        std::string model_path;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::string name;
        TF_HANDLE * handler;
        InferenceFrozenGraph_c();
        InferenceFrozenGraph_c(const std::string & model_path,const std::vector<std::string> &  input_names,
        const std::vector<std::string> & output_names,const std::string & name);
        void print_info();
        TF_HANDLE * load_graph();
        void inference(std::vector<float> & image,
             int height, int width, int channel,std::vector<std::vector<float>> & result_data,
             std::vector<std::vector<int>> & dims);
        ~InferenceFrozenGraph_c();
    };
}
#endif
```

- InferenceFrozenGraph.cpp

```c++
#include <iostream>
#include "InferenceFrozenGraph.h"
#include <fstream>
#include <cstdlib>
#include <typeinfo>

static TF_TENSOR tensorOf(TF_DATA_TYPE type, int height, int width, int channel, int batch=1) {
    TF_TENSOR tensor;
    tensor.type = type;
    tensor.dims[0] = batch;
    tensor.dims[1] = height;
    tensor.dims[2] = width;
    tensor.dims[3] = channel;
    return tensor;
}

char* readBuffer(const char* filename, int* fsize) {
  std::ifstream f(filename, std::ios::binary);
  if (f.fail() || !f.is_open()) {
    return nullptr;
  }
  f.seekg(0, std::ios::end);
  *fsize = f.tellg();    //TF_Buffer.lenght
  f.seekg(0, std::ios::beg);
  if (*fsize < 1) {
    f.close();
    return nullptr;
  }

  auto data = static_cast<char*>(std::malloc(*fsize));
  f.read(data, *fsize);       //TF_Buffer.data
  f.close();
  return data;
}

namespace inference {

    // Default constructor
    InferenceFrozenGraph_c::InferenceFrozenGraph_c () {}

    // Overloaded constructor
    InferenceFrozenGraph_c::InferenceFrozenGraph_c(const std::string & model_path,const std::vector<std::string>& input_names,
        const std::vector<std::string>& output_names, const std::string & name){
        this->model_path = model_path;
        this->name = name;
        for (auto p = input_names.begin(); p!=input_names.end();++p){
            this->input_names.push_back(*p);
        }
        for (auto p = output_names.begin(); p!=output_names.end();++p){
            this->output_names.push_back(*p);
        }

        this->handler = this->load_graph();
    }

    // Destructor
    InferenceFrozenGraph_c::~InferenceFrozenGraph_c () {}


    // Load graph
    TF_HANDLE * InferenceFrozenGraph_c::load_graph(){
        int fsize = -1;
        char* buffer = readBuffer(this->model_path.c_str(), &fsize);
        if (fsize < 0 || !buffer) {
            std::cout << "Error: ReadBufferFromFile" << std::endl;
        }

        int device = -1;

        const int input_count = this->input_names.size();
        const int output_count = this->output_names.size();


        const char **inputs_names = new const char*[input_count];
        for (int i = 0; i < input_count; ++i)
            inputs_names[i] = this->input_names[i].c_str();

        const char **output_names = new const char*[output_count];
        for (int i = 0; i < output_count; ++i)
            output_names[i] = this->output_names[i].c_str();

        TF_HANDLE *handler = tf_create2(buffer, fsize, device, inputs_names, input_count, output_names, output_count);
        if (!handler) {
            std::cout << "handler is NULL " << std::endl;
        }
        std::cout << ">>> handler complete" << std::endl;

        delete [] inputs_names;
        delete [] output_names;

//        free(buffer);

        return handler;
    }


     void InferenceFrozenGraph_c::inference(std::vector<float> & image,
             int height, int width, int channel,std::vector<std::vector<float>> & result_data,
             std::vector<std::vector<int>> & dims){

        TF_TENSOR image_tensor = tensorOf(TF_F32, height, width, channel) ;

        image_tensor.data = image.data();

        TF_TENSOR inputs[1] = {image_tensor};

        const int output_count = this->output_names.size();

        std::vector<TF_TENSOR> outputs(output_count);

//        std::cout << "tf_infer start" << std::endl;
        tf_infer(handler, inputs, outputs.data());
//        std::cout << "tf_infer complete" << std::endl;

        std::vector<float> result;
        std::vector<int> dim;

        for(int i = 0; i < output_count; ++i) {
            result.clear();
            dim.clear();
            const auto ot = (float*) outputs[i].data;

            for(int j=0; j < outputs[i].len() ; ++j) {

                result.push_back(ot[j]);
            }

            for (int j=0; j<4;++j){
                dim.push_back(outputs[i].dims[j]);
            }

            result_data.push_back(result);
            dims.push_back(dim);
        }
    }

    // Print info
    void InferenceFrozenGraph_c::print_info () {
        std::cout<<this->model_path<<std::endl;
        std::cout<<this->name<<std::endl;
        std::cout<<"input_names:"<<std::endl;
        for (auto p = this->input_names.begin(); p!=this->input_names.end();++p){
            std::cout<<"  "<<*p<<std::endl;
        }
        std::cout<<"output_names:"<<std::endl;
        for (auto p = this->output_names.begin(); p!=this->output_names.end();++p){
            std::cout<<"  "<<*p<<std::endl;
        }
    }

}

int main(int argc, char const *argv[])
{
    std::string s1 = "path_to_model";
    std::string name = "name";
    std::vector<std::string> v1;
    std::vector<std::string> v2;
    v1.push_back("image");
    v2.push_back("class");
    v2.push_back("probility");
    inference::InferenceFrozenGraph_c* infer = new inference::InferenceFrozenGraph_c(s1,v1,v2,name);
    infer->print_info();
    return 0;
}

```

- InferenceFrozenGraph.pxd

```python
# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "InferenceFrozenGraph.cpp":
    pass

# Declare the class with cdef
cdef extern from "InferenceFrozenGraph.h" namespace "inference":


    cdef cppclass InferenceFrozenGraph_c:
        InferenceFrozenGraph_c() except +
        InferenceFrozenGraph_c(string, vector[string], vector[string], string) except +
        string model_path
        vector[string] input_names
        vector[string] output_names
        string name

        void print_info()
        void inference(vector[float],int,int,int,vector[vector[float]],vector[vector[int]])

        # TF_HANDLE * load_graph()

```

- infer_module.pyx

```python
# distutils: language = c++
# cython: language_level=3

from InferenceFrozenGraph cimport InferenceFrozenGraph_c
from libcpp.string cimport string
from libcpp.vector cimport vector
import numpy as np

cdef class InferenceFrozenGraph:
    cdef InferenceFrozenGraph_c c_infer

    def __init__(self,str model_path,str input_name, list output_names, str name,graph=None,str gpus=''):
        cdef string s1 = <string> model_path.encode('utf-8')
        cdef string s2 = <string> "".encode('utf-8')
        cdef vector[string] input
        cdef vector[string] output
        input.push_back(input_name.encode('utf-8'))
        for i in output_names:
            # i = name + '/' + i
            output.push_back(i.encode('utf-8'))
        self.c_infer = InferenceFrozenGraph_c(s1,input,output,s2)

    def print_info(self):
        self.c_infer.print_info()

    def __call__(self, image):
        image = image[0]
        cdef int height = image.shape[0]
        cdef int width = image.shape[1]
        cdef int channel = image.shape[2]
        cdef vector[float]  image_faltten
        cdef vector[vector[float]] results
        cdef vector[vector[int]] dims

        image_faltten = image.flatten()
        self.c_infer.inference(image_faltten,height,width,channel,results,dims)

        res = []
        for i in range(len(results)):
            dim_shape = [d for d in dims[i] if d!=0]
            r = np.array(results[i],dtype=np.float32).reshape(dim_shape)
            res.append(r[:])
        return res
```

- setup.py

```python
# cython: language_level=3
import sys
from setuptools import setup,Extension
from Cython.Build import cythonize

ext_module = Extension(
    "infer_module",
    sources=["infer_module.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"],
    include_dirs=["/home/wangjingwen/projects_2019/OCR/tel_ocr/inferenceImage/include"],
    libraries    = ["tfeasy"],
    library_dirs = ["/home/wangjingwen/projects_2019/OCR/tel_ocr/inferenceImage/tfeasy/centos"]
)
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('build_ext')
        sys.argv.append('--inplace')

    setup(ext_modules=cythonize(ext_module))
```



注意`__cinit__`和`__init__`之间的区别：

https://stackoverflow.com/questions/18260095/cant-override-init-of-class-from-cython-extension