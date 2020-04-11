#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#define BLOCK_SIZE 16

#define		NROW	1024
#define		NCOL	NROW

#define TEST_RESULTS

using namespace std;

//Input Array A
int inputArrayA  [NROW][NCOL];
//Input Array B
int inputArrayB  [NROW][NCOL];
//Output Array C
int outputArrayC [NROW][NCOL];


void mmult()
{
    for (int i = 0; i < NROW; ++i)
        for (int j = 0; j < NCOL; ++j)
            for (int k = 0; k < NROW; ++k )
		outputArrayC[i][j] += inputArrayA[i][k] * inputArrayB[k][j] * 2;
}


__global__ void mmult_kernel(const int * a, const int * b, int * c)
{
	int globx = blockIdx.x * blockDim.x + threadIdx.x;
	int globy = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int i;

	for (i = 0; i < NROW; i++)
		c[globx * NROW + globy] += a[globx * NROW + i] * b[i * NROW + globy] * 2;
}

void mmult_gpu(const int * a, const int * b, int * c)
{	
	dim3 dim_Grid(NROW/BLOCK_SIZE, NCOL/BLOCK_SIZE);
	dim3 dim_Block(BLOCK_SIZE,BLOCK_SIZE);
	mmult_kernel<<<dim_Grid, dim_Block>>>(a, b, c);
}

int main()
{

    int * a, * b, * c, * a_gpu, * b_gpu, * c_gpu;

    a = new int[NROW * NCOL];
    b = new int[NROW * NCOL];
    c = new int[NROW * NCOL];

    cudaMalloc((void **) &a_gpu, NROW * NCOL * sizeof *a_gpu);
    cudaMalloc((void **) &b_gpu, NROW * NCOL * sizeof *b_gpu);
    cudaMalloc((void **) &c_gpu, NROW * NCOL * sizeof *c_gpu);

    for (int i = 0; i < NROW; ++i)
	for( int j = 0; j < NCOL; j++)
          a[i*NROW +j] = i * NROW + j;

    for (int i = 0; i < NROW; ++i)
        for( int j = 0; j < NCOL; j++)
	  b[i*NROW + j] = j * NROW + j;
 

    for(int i=0;i<NROW;i++){
	for(int j=0;j<NCOL;j++){
	    inputArrayA[i][j]= i*NCOL+j;
	    inputArrayB[i][j]= j*NCOL+j;
	    outputArrayC[i][j] = 0;
	}
    }

    
    cudaMemcpy(a_gpu, a, NROW * NCOL * sizeof *a_gpu, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, NROW * NCOL * sizeof *b_gpu, cudaMemcpyHostToDevice);


    for(int i=0;i<NROW;i++) {
	for(int j=0;j<NCOL;j++) {
	    inputArrayA[i][j]= i*NCOL+j;
	    inputArrayB[i][j]= j*NCOL+j;
	}
    }

    float sTime;
    clock_t start, finish;
 
   
    start = clock();
    mmult();
    finish = clock();
    
    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on CPU: %lf sec\n", sTime);


    start = clock();
    mmult_gpu(a_gpu, b_gpu, c_gpu);
    cudaThreadSynchronize();
    cudaMemcpy(c, c_gpu, NROW * NCOL * sizeof *c_gpu, cudaMemcpyDeviceToHost);
    finish = clock();

    sTime = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Run time on GPU: %lf sec\n",sTime);

    double totalSum_cpu;

    for (int i = 0; i < NROW; ++i)
	for(int j = 0; j < NCOL; j++)
            totalSum_cpu += (double)outputArrayC[i][j];
 
    std::cout << "totalSum_cpu = " << totalSum_cpu << std::endl;

    double totalSum_gpu;

    for (int i = 0; i < NROW * NCOL; ++i)
        totalSum_gpu += (double)c[i];

    std::cout << "totalSum_gpu = " << totalSum_gpu << std::endl;

    cudaFree(c_gpu);
    cudaFree(b_gpu);
    cudaFree(a_gpu);
    
    delete [] c;
    delete [] b;
    delete [] a;
}
