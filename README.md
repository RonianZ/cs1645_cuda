# cs1645_cuda

how to compile and run:

nvcc -g -Wno-deprecated-gpu-targets mm.cu -lcudart -o mm
nvcc -g -Wno-deprecated-gpu-targets rect.cu -lcudart -o rect

./mm
./rect

The programs are tested in "cuda 8.0.44"
