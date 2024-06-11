# Uisng cython to build .so

In case you use virtual environment from project root directory, e.g. in directory `venv`:  
```sh
virtualenv venv
source venv/bin/activate
```

## Install cython

You can pip install cython or just run:  
```sh
pip3 install -r requirements.txt
```

## Build the .so
```sh
cd so
python3 setup_classifier.py build_ext --inplace && mv *.so ../
```
Example:  
```sh
(venv) ubuntu@ubuntu:~/py/github.com/seblkma/ml/py-ml/04_binary_classification_so$ cd so
(venv) ubuntu@ubuntu:~/py/github.com/seblkma/ml/py-ml/04_binary_classification_so/so$ python3 setup_classifier.py build_ext --inplace && mv *.so ../
/home/ubuntu/py/github.com/seblkma/ml/py-ml/04_binary_classification_so/so/setup_classifier.py:2: DeprecationWarning: The distutils package is deprecated and slated for removal in Python 3.12. Use setuptools or check PEP 632 for potential alternatives
  from distutils.core import setup
running build_ext
building 'binary_classifier' extension
aarch64-linux-gnu-gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/home/ubuntu/py/github.com/seblkma/ml/py-ml/venv/include -I/usr/include/python3.10 -c binary_classifier.c -o build/temp.linux-aarch64-3.10/binary_classifier.o
aarch64-linux-gnu-gcc -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 build/temp.linux-aarch64-3.10/binary_classifier.o -o /home/ubuntu/py/github.com/seblkma/ml/py-ml/04_binary_classification_so/so/binary_classifier.so
```

## Run your main python script
Example:  
```sh
python3 classifier.py
```
Output:  
```sh
...
Iteration 9996 => Loss: 0.36573204385028551533
Iteration 9997 => Loss: 0.36573094483949619704
Iteration 9998 => Loss: 0.36572984584704604227
Iteration 9999 => Loss: 0.36572874687292949991

The transposed Weights: [[ Bias, Reservations, Temperature, Tourists]]
               Weights: [[-0.37450392  0.51754011 -0.35263466  0.25625742]]

Success: 25/30 (83.33%)
```