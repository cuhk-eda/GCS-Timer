# GCS-Timer

GCS-Timer is a GPU-accelerated static timing analyzer based on the advanced Composite Current Source (CCS) timing model. 

Check out the following paper for more details.

* Shiju Lin, Guannan Guo, Tsung-Wei Huang, Weihua Sheng, Evangeline F.Y. Young and Martin D.F. Wong, ["GCS-Timer: GPU-Accelerated Current Source Model Based Static Timing Analysis"](https://shijulin.github.io/files/430_Camera_Ready_Paper.pdf), ACM/IEEE Design Automation Conference (DAC), San Francisco, CA, USA, June 23–27, 2024.

## Compile
```bash
nvcc src/main.cpp -x cu -arch=sm_86 -o GCS_Timer -std=c++14 -O3
```
You may want to change `-arch=sm_86` according to your GPU.

## Run
```bash
./GCS_Timer <benchmark_name>
```
`benchmark_name` can be `mul`, `log2`, `div` and `hyp`, which are the designs used for evaluation in the paper. 

The spef files of `div` and `hyp` are zipped and need to be unzipped first.

## Contact
[Shiju Lin](shijulin.github.io) (email: sjlin@cse.cuhk.edu.hk)

## License
BSD 3-Clause License
