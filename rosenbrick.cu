#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "constants.h"

// assume block size equal population size

#define sqr(x) (x)*(x)

void cudasafe(cudaError_t error, char* message = "Error occured") {
	if(error != cudaSuccess) {
		fprintf(stderr,"ERROR: %s : %i\n", message, error);
		exit(-1);
	}
}

struct ScoreWithId {
	float score;
	int id;
};

__global__ void randomInit(curandState* state, unsigned long seed) {
    int tid = threadIdx.x;
    curand_init(seed, tid, 0, state + tid);
}

__global__ void calcScore(const float* population, ScoreWithId* score) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < POPULATION_SIZE) {
		float result = 0;
	
		const float *curPos = &population[tid * VAR_NUMBER];
		for (size_t i=0; i<VAR_NUMBER-1; ++i) {
			result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
			++curPos;
		}

		score[tid].score = result;
		score[tid].id = tid;
	}
}

struct ScoreCompare {
	__host__ __device__ bool operator() (const ScoreWithId& a, const ScoreWithId& b) const {
		return a.score < b.score;
	}
};

__global__ void produceGeneration(const float* population, float* nextGeneration, const ScoreWithId* score, curandState* randomStates) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	float* nextGenerationPos = &nextGeneration[tid * VAR_NUMBER];
	const float* individual = &population[score[tid % (POPULATION_SIZE / 3)].id * VAR_NUMBER];

	if (tid < POPULATION_SIZE / 3) { // copy as is
		for (int i=0; i<VAR_NUMBER; ++i) {
			*nextGenerationPos = *individual;
			++nextGenerationPos;
			++individual;
		}
	} else {
		curandState &localState = randomStates[threadIdx.x];
		if (tid < POPULATION_SIZE * 2 / 3) { // mutate
			for (int i=0; i<VAR_NUMBER; ++i) {
				*nextGenerationPos = *individual + powf(10.0, ((curand_uniform(&localState) * 17) - 15)) * (curand_uniform(&localState) < 0.5f ? -1 : 1);
				++nextGenerationPos;
				++individual;
			}
		} else if (tid < POPULATION_SIZE) { // crossover
			const int otherIndividualIndex = (tid + static_cast<int>(curand_uniform(&localState) * POPULATION_SIZE)) % (POPULATION_SIZE / 3);
			const float* otherIndividual = &population[score[otherIndividualIndex].id * VAR_NUMBER];

			for (int i=0; i<VAR_NUMBER; ++i) {
				*nextGenerationPos = (*individual + *otherIndividual) * 0.5f;
				++nextGenerationPos;
				++individual;
				++otherIndividual;
			}
		}
	}
}

double solveGPU() {
	double ans = 0;

	const int MAX_THREADS_PER_BLOCK = 512;

	ScoreWithId score[POPULATION_SIZE];
	float *population = new float[POPULATION_SIZE * VAR_NUMBER];

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i * VAR_NUMBER + u] = float_random();
		}
	}

	// copying population to device
	float *devicePopulation = 0;
	float *nextGeneration = 0;
	ScoreWithId *deviceScore = 0;
	curandState* randomStates;

	cudasafe(cudaMalloc(&randomStates, MAX_THREADS_PER_BLOCK * sizeof(curandState)), "Could not allocate memory for randomStates");
	cudasafe(cudaMalloc((void **)&devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for devicePopulation");
	cudasafe(cudaMalloc((void **)&nextGeneration, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for nextGeneration");
	cudasafe(cudaMalloc((void **)&deviceScore, POPULATION_SIZE * sizeof (ScoreWithId)), "Could not allocate memory for deviceScore");

	thrust::device_ptr<ScoreWithId> deviceScorePtrBegin(deviceScore);
	thrust::device_ptr<ScoreWithId> deviceScorePtrEnd = deviceScorePtrBegin + POPULATION_SIZE;

	cudasafe(cudaMemcpy(devicePopulation, population, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyHostToDevice), "Could not copy population to device");

	// invoking random init
	randomInit<<<1, MAX_THREADS_PER_BLOCK>>>(randomStates, 900);
	cudasafe(cudaGetLastError(), "Could not invoke kernel randomInit");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling randomInit");

	const int BLOCKS_NUMBER = (POPULATION_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	for (int i=0; i<30000; i++) {
		// invoking calcScore
		calcScore<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(devicePopulation, deviceScore);
		cudasafe(cudaGetLastError(), "Could not invoke kernel calcScore");
		cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calsScore");

		thrust::sort(deviceScorePtrBegin, deviceScorePtrEnd, ScoreCompare());

		produceGeneration<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(devicePopulation, nextGeneration, deviceScore, randomStates);
		cudasafe(cudaGetLastError(), "Could not invoke kernel produceGeneration");
		cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after produceGeneration");

		std::swap(devicePopulation, nextGeneration);

		std::cout << "printing first 10 elements of score:" << std::endl;
		cudasafe(cudaMemcpy(score, deviceScore, POPULATION_SIZE * sizeof (ScoreWithId), cudaMemcpyDeviceToHost), "Could not copy score to host");
		for (int i=0; i<10; i++)
			std::cout << score[i].score << ' ';
		std::cout << std::endl;
	}

	// freeing memory
	cudasafe(cudaFree(devicePopulation), "Failed to free devicePopulation");
	cudasafe(cudaFree(deviceScore), "Failed to free deviceScore");
	cudasafe(cudaFree(randomStates), "Could not free randomStates");
	cudasafe(cudaFree(nextGeneration), "Could not free nextGeneration");

	delete[] population;

	return ans;
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	double ans = solveGPU();
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}