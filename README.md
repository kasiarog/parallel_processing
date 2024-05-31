# Parallel processing using OpenMP and CUDA
## Problem
The problem of removing communication routes in the city

# Compilation
## iterative C++
```
g++ iterative\main.cpp -o iterative\main

```

## parallel OpenMP
```
g++ openmp\openmp.cpp -o openmp\openmp -fopenmp
```

## parallel CUDA
```
nvcc cuda\cuda.cu -o cuda\cuda -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\bin"
```
