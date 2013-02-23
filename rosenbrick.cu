#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>

#include <thrust/sort.h>

#include "constants.h"

// assume block size equal population size

#define sqr(x) (x)*(x)

void cudasafe(cudaError_t error, char* message = "Error occured")
{
	if(error != cudaSuccess) {
		fprintf(stderr,"ERROR: %s : %i\n", message, error);
		exit(-1);
	}
}

__global__ void calcScore(const float* population, float* score) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < POPULATION_SIZE) {
		float result = 0;
	
		const float *curPos = &population[tid * VAR_NUMBER];
		for (size_t i=0; i<VAR_NUMBER-1; ++i) {
			result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
			++curPos;
		}

		score[tid] = result;
	}
}

double solveGPU() {
	double ans = 0;

	float score[POPULATION_SIZE];
	float population[POPULATION_SIZE][VAR_NUMBER];
	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i][u] = float_random();
		}
	}

	// copying population to device
	float *devicePopulation = 0;
	float *deviceScore = 0;

	cudasafe(cudaMalloc((void **)&devicePopulation, sizeof population), "Could not allocate memory for devicePopulation");
	cudasafe(cudaMalloc((void **)&deviceScore, POPULATION_SIZE * sizeof (float)), "Could not allocate memory for deviceScore");

	cudasafe(cudaMemcpy(devicePopulation, population, sizeof population, cudaMemcpyHostToDevice), "Could not copy population to device");

	// invoking calcScore

	const int MAX_THREADS_PER_BLOCK = 512;
	const int BLOCKS_NUMBER = (POPULATION_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	calcScore<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(devicePopulation, deviceScore);
	cudasafe(cudaGetLastError(), "Could not invoke kernel calcScore");

	cudaDeviceSynchronize();

	thrust::device_ptr<float> deviceScorePtrBegin(deviceScore);
	thrust::device_ptr<float> deviceScorePtrEnd = deviceScorePtrBegin + POPULATION_SIZE;

	thrust::sort(deviceScorePtrBegin, deviceScorePtrEnd);

	cudasafe(cudaMemcpy(score, deviceScore, sizeof score, cudaMemcpyDeviceToHost), "Could not copy score to host");
	//std::sort(score, score + POPULATION_SIZE);
	for (int i=0; i<100; i++)
		std::cout << score[i] << std::endl;

	// freeing memory
	cudasafe(cudaFree(devicePopulation), "Failed to free devicePopulation");
	cudasafe(cudaFree(deviceScore), "Failed to free deviceScore");

	return ans;
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	double ans = solveGPU();
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}