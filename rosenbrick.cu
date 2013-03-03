#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

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

__global__ void calcScore(const float* population, float* score, float* scoreTmp) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < POPULATION_SIZE) {
		float result = 0;
	
		const float *curPos = &population[tid * VAR_NUMBER];
		for (size_t i=0; i<VAR_NUMBER-1; ++i) {
			result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
			++curPos;
		}

		score[tid] = scoreTmp[tid] = result;
	}
}

__global__ void produceGeneration(const float* population, const float* nextGeneration, const float* score, const float* limitScorePtr) {
	const float limitScore = *limitScorePtr;


}

double solveGPU() {
	double ans = 0;

	float score[POPULATION_SIZE];
	float *population = new float[POPULATION_SIZE * VAR_NUMBER];
	float *nextGeneration = new float[POPULATION_SIZE * VAR_NUMBER];

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i * VAR_NUMBER + u] = float_random();
		}
	}

	// copying population to device
	float *devicePopulation = 0;
	float *deviceScore = 0;
	float *deviceScoreTmp = 0;

	cudasafe(cudaMalloc((void **)&devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for devicePopulation");
	cudasafe(cudaMalloc((void **)&deviceScore, POPULATION_SIZE * sizeof (float)), "Could not allocate memory for deviceScore");
	cudasafe(cudaMalloc((void **)&deviceScoreTmp, POPULATION_SIZE * sizeof (float)), "Could not allocate memory for deviceScoreTmp");

	thrust::device_ptr<float> deviceScorePtrBegin(deviceScoreTmp);
	thrust::device_ptr<float> deviceScorePtrEnd = deviceScorePtrBegin + POPULATION_SIZE;

	cudasafe(cudaMemcpy(devicePopulation, population, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyHostToDevice), "Could not copy population to device");

	// invoking calcScore
	const int MAX_THREADS_PER_BLOCK = 512;
	const int BLOCKS_NUMBER = (POPULATION_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	calcScore<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(devicePopulation, deviceScore, deviceScoreTmp);
	cudasafe(cudaGetLastError(), "Could not invoke kernel calcScore");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device");

	thrust::sort(deviceScorePtrBegin, deviceScorePtrEnd);

	produceGeneration<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(population, nextGeneration, deviceScore, deviceScoreTmp + POPULATION_SIZE / 3);
	cudasafe(cudaGetLastError(), "Could not invoke kernel produce nextGeneration");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device");

	std::cout << "printing first 10 elements of score:" << std::endl;
	cudasafe(cudaMemcpy(score, deviceScoreTmp, sizeof score, cudaMemcpyDeviceToHost), "Could not copy score to host");
	for (int i=0; i<10; i++)
		std::cout << score[i] << ' ';
	std::cout << std::endl;

	// freeing memory
	cudasafe(cudaFree(devicePopulation), "Failed to free devicePopulation");
	cudasafe(cudaFree(deviceScore), "Failed to free deviceScore");
	cudasafe(cudaFree(deviceScoreTmp), "Failed to free deviceScoreTmp");

	delete[] population;
	delete[] nextGeneration;

	return ans;
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	double ans = solveGPU();
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}