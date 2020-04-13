#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <cuda_runtime_api.h>

#define		NSTEPS	8388600
#define		NITER 	8388600
#define		P_START	0 
#define		P_END	10 


struct timeval startTime;
struct timeval finishTime;
double timeIntervalLength;



__global__
void rect(int n, double h, double *area)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  double p_current;
  double f_result;

  for (int i = index; i < n; i += stride) {

    p_current = i*h;
    f_result = cos(p_current);
    area[i] += f_result*h;
  }
}

int main(int argc, char* argv[])
{

	int i;
	int blockSize, numBlocks;
	double h;
	double area_cpu;
	double area_gpu;
	double *area_gpu_device;
	double *area_gpu_host;
	double p_current = P_START;
	double f_result;

	cudaMalloc((void **) &area_gpu_device, NITER * sizeof *area_gpu_device);
	area_gpu_host = (double*) malloc(NITER * sizeof(double));
	//
	//I N I T I A L I Z A T I O N S
	//
	h = (double)(P_END-P_START)/NSTEPS;
	p_current = P_START;
	area_cpu=0.0;


	//Get the start time
	gettimeofday(&startTime, NULL);


	for(i = 0; i < NITER; i++)
	{
		p_current = i*h;
		f_result = cos(p_current);
		area_cpu += f_result*h;
		p_current += h;
	}

	//Get the end time
	gettimeofday(&finishTime, NULL);  /* after time */
	
	
	
	
	//Calculate the interval length 
	timeIntervalLength = (double)(finishTime.tv_sec-startTime.tv_sec) * 1000000 
		     + (double)(finishTime.tv_usec-startTime.tv_usec);
	timeIntervalLength=timeIntervalLength/1000;

	//Print the interval lenght
	printf("Interval length on CPU: %g msec.\n", timeIntervalLength);

        printf("Result on GPU: %.2lf \n",area_cpu);

	
	blockSize = 256;
	numBlocks = (NITER + blockSize - 1) / blockSize;

	//Get the start time
	gettimeofday(&startTime, NULL);

	rect<<<numBlocks, blockSize>>>(NITER, h, area_gpu_device);


	cudaThreadSynchronize();
	cudaMemcpy(area_gpu_host, area_gpu_device, NITER * sizeof *area_gpu_device, cudaMemcpyDeviceToHost);
	for (int i = 0; i < NITER; ++i)
        	area_gpu += area_gpu_host[i];

	//Get the end time
	gettimeofday(&finishTime, NULL);  /* after time */

	
	//Calculate the interval length 
	timeIntervalLength = (double)(finishTime.tv_sec-startTime.tv_sec) * 1000000 
		     + (double)(finishTime.tv_usec-startTime.tv_usec);
	timeIntervalLength=timeIntervalLength/1000;

	//Print the interval lenght
	printf("Interval length on GPU: %g msec.\n", timeIntervalLength);

        printf("Result on GPU: %.2lf \n",area_gpu);

	return 0;
}
