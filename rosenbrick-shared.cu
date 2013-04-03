#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "constants.h"

// assume block size equal population size

template <class T>
__device__ inline T sqr(const T& value) {
	return value * value;
}

const int MAX_THREADS_PER_BLOCK = 128;

void cudasafe(cudaError_t error, char* message = "Error occured") {
	if(error != cudaSuccess) {
		fprintf(stderr,"ERROR: %s : %i\n", message, error);
		exit(-1);
	}
}

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
	float signs[2] = {-1.0f, 1.0f};
	
	float* nextGenerationPos = &nextGeneration[tid * VAR_NUMBER];
	const float* individual = &population[score[tid % (POPULATION_SIZE / 3)].id * VAR_NUMBER];
	//const float* individual = &population[score[tid].id * VAR_NUMBER];

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
				const float sign = signs[static_cast<int>(curand_uniform(&localState)*2)];
				*nextGenerationPos = *individual + powf(10.0, ((curand_uniform(&localState) * 17) - 15)) * sign;
				++nextGenerationPos;
				++individual;
			}
		} else if (tid < POPULATION_SIZE) { // crossover
			const int otherIndividualIndex = (tid + static_cast<int>(curand_uniform(&localState) * POPULATION_SIZE)) % (POPULATION_SIZE / 3);
			const float* otherIndividual = &population[otherIndividualIndex * VAR_NUMBER];

			for (int i=0; i<VAR_NUMBER; ++i) {
				*nextGenerationPos = (*individual + *otherIndividual) * 0.5f;
				++nextGenerationPos;
				++individual;
				++otherIndividual;
			}
		}
	}
}

__global__ void GAKernel(float* population, ScoreWithId* score, curandState* randomStates) {
	__shared__ float sharedPopulation[MAX_THREADS_PER_BLOCK][VAR_NUMBER];
	__shared__ float sharedScore[MAX_THREADS_PER_BLOCK];
	const float signs[2] = {-1.0f, 1.0f};

	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid = threadIdx.x;

	// loading initial random population into shared memory
    if (gid < POPULATION_SIZE) {
        for (int i=0; i<VAR_NUMBER; ++i)
            sharedPopulation[tid][i] = population[gid * VAR_NUMBER + i];
    }

	for (int i=0; i<VAR_NUMBER; ++i)
		sharedPopulation[tid][i] = i + tid;

	curandState &localState = randomStates[tid];
	for (int generationIndex=0; ; ++generationIndex) {
		__syncthreads();

		// calculating score
		const float *curPos = sharedPopulation[tid];
		float result = 0;
		for (size_t i=0; i<VAR_NUMBER-1; ++i) {
			result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
			++curPos;
		}
		sharedScore[tid] = result;

		__syncthreads();

		if (generationIndex == 129990) break;

		// selection

		if (tid < MAX_THREADS_PER_BLOCK / 2) {
			// first half of threads writes best individual into its position
			if (sharedScore[tid] > sharedScore[tid + MAX_THREADS_PER_BLOCK / 2]) {
				for (int i=0; i<VAR_NUMBER; ++i)
					sharedPopulation[tid][i] = sharedPopulation[tid + MAX_THREADS_PER_BLOCK / 2][i];
			}
		}

		__syncthreads();

		// now we've got best individuals in the first half of sharedPopulation

		// crossovers
		if (tid > MAX_THREADS_PER_BLOCK / 2) {
			int first = curand_uniform(&localState) * (MAX_THREADS_PER_BLOCK / 2);
			int second = curand_uniform(&localState) * (MAX_THREADS_PER_BLOCK / 2);
		
			// TODO: implement weight here?
			for (int i=0; i<VAR_NUMBER; ++i) {
				sharedPopulation[tid][i] = (sharedPopulation[first][i] + sharedPopulation[second][i]) * 0.5f;
			}
		}

		__syncthreads();

		// mutations
        if (tid > MAX_THREADS_PER_BLOCK / 2) {
            if (curand_uniform(&localState) < 0.8) {
                for (int i=0; i<VAR_NUMBER; ++i) {
                    const float sign = signs[static_cast<int>(curand_uniform(&localState)*2)];
                    sharedPopulation[tid][i] += powf(10.0, ((curand_uniform(&localState) * 17) - 15)) * sign;
                }
            }
        }
	}

	// output current population back
    if (gid < POPULATION_SIZE) {
        for (int i=0; i<VAR_NUMBER; ++i)
            population[gid * VAR_NUMBER + i] = sharedPopulation[tid][i];

        score[gid].score = sharedScore[tid];
    }
}

void printPopulation(const float* devicePopulation, const ScoreWithId* deviceScore) {
	float population[POPULATION_SIZE][VAR_NUMBER];
	cudasafe(cudaMemcpy(population, devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyDeviceToHost), "Could not copy population from device");

	ScoreWithId score[POPULATION_SIZE];
	cudasafe(cudaMemcpy(score, deviceScore, POPULATION_SIZE * sizeof (ScoreWithId), cudaMemcpyDeviceToHost), "Could not copy score to host");

	//std::cout.cetf(std::ios::fixed);
	std::cout.precision(12);
	
	for (int i=0; i<POPULATION_SIZE; ++i) {
		std::cout << std::setw(15) << i << ' ';
	}
	std::cout << std::endl;

	for (int i=0; i<VAR_NUMBER; i++) {
		for (int u=0; u<POPULATION_SIZE; ++u) {
			std::cout << std::setw(15) << population[u][i] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << "Score: " << std::endl;
	for (int i=0; i<POPULATION_SIZE; ++i) {
		std::cout << std::setw(15) << score[i].score << ' ';
	}
	std::cout << std::endl;
}

double solveGPU() {
	double ans = 0;

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

	cudasafe(cudaMemcpy(devicePopulation, population, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyHostToDevice), "Could not copy population to device");

	// invoking random init
	randomInit<<<1, MAX_THREADS_PER_BLOCK>>>(randomStates, 900);
	cudasafe(cudaGetLastError(), "Could not invoke kernel randomInit");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling randomInit");

	const int BLOCKS_NUMBER = (POPULATION_SIZE + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
	GAKernel<<<BLOCKS_NUMBER, MAX_THREADS_PER_BLOCK>>>(devicePopulation, deviceScore, randomStates);
	cudasafe(cudaGetLastError(), "Could not invoke GAKernel");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling GAKernel");

	printPopulation(devicePopulation, deviceScore);

	// freeing memory
	cudasafe(cudaFree(devicePopulation), "Failed to free devicePopulation");
	cudasafe(cudaFree(deviceScore), "Failed to free deviceScore");
	cudasafe(cudaFree(randomStates), "Could not free randomStates");
	cudasafe(cudaFree(nextGeneration), "Could not free nextGeneration");

	delete[] population;

	return ans;
}

int main() {
	freopen("output.txt", "w", stdout);
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	double ans = solveGPU();
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}
