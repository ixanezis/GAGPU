#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>
#include <iomanip>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "constants.h"

// assume block size equal population size

template <class T>
__device__ inline T sqr(const T& value) {
	return value * value;
}

const int THREADS_PER_BLOCK = 256;

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

__device__ float rosenbrock(const float* curPos) {
    float result = 0;
    for (size_t i=0; i<VAR_NUMBER-1; ++i) {
        result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
        ++curPos;
    }
    return result;
}

__device__ float rastrigin(const float *curPos) {
    float result = 10.0f * VAR_NUMBER;
    for (size_t i=0; i<VAR_NUMBER; ++i) {
        result += *curPos * *curPos - 10.0f * cosf(2 * CUDART_PI_F * *curPos);
        ++curPos;
    }
    return result;
}

__global__ void GAKernel(float* population, ScoreWithId* score, curandState* randomStates) {
	__shared__ float sharedPopulation[THREADS_PER_BLOCK * 2][VAR_NUMBER];
	__shared__ float sharedScore[THREADS_PER_BLOCK * 2];
	const float SIGN[2] = {-1.0f, 1.0f};
    const float MULT[2] = {1.0f, 0.0f};

	const int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const int tid = threadIdx.x;

	// loading initial random population into shared memory
    if (gid < POPULATION_SIZE) {
        for (int i=0; i<VAR_NUMBER; ++i)
            sharedPopulation[tid][i] = population[gid * VAR_NUMBER + i];
    }

    sharedScore[tid + THREADS_PER_BLOCK] = 123123.0;

    __syncthreads();
    // we first have to calculate the score for the first half of threads
    const float *curPos = sharedPopulation[tid];
    sharedScore[tid] = rosenbrock(curPos);

	curandState &localState = randomStates[tid];
	for (int generationIndex=0; ; ++generationIndex) {
		__syncthreads();

		// calculating score for the second half of individuals
		const float *curPos = sharedPopulation[tid + THREADS_PER_BLOCK];
		sharedScore[tid + THREADS_PER_BLOCK] = rosenbrock(curPos);

		__syncthreads();

		if (generationIndex == 400000) break;

		// selection
        // first half of threads writes best individual into its position
        if (sharedScore[tid] > sharedScore[tid + THREADS_PER_BLOCK]) {
            for (int i=0; i<VAR_NUMBER; ++i)
                sharedPopulation[tid][i] = sharedPopulation[tid + THREADS_PER_BLOCK][i];
            sharedScore[tid] = sharedScore[tid + THREADS_PER_BLOCK];
        }

		__syncthreads();

		// now we've got best individuals in the first half of sharedPopulation

		// crossovers
        const int first = curand_uniform(&localState) * THREADS_PER_BLOCK;
        const int second = curand_uniform(&localState) * THREADS_PER_BLOCK;
    
        const float weight = curand_uniform(&localState);
        for (int i=0; i<VAR_NUMBER; ++i) {
            sharedPopulation[tid + THREADS_PER_BLOCK][i] = sharedPopulation[first][i] * weight + sharedPopulation[second][i] * (1.0f - weight);
        }

		__syncthreads();

		// mutations on second half of population
        if (curand_uniform(&localState) < 0.8) {
            const float order = (curand_uniform(&localState) * 17) - 15;
            for (int i=0; i<VAR_NUMBER; ++i) {
                const float mult = MULT[curand_uniform(&localState) < 0.8f];
                const float sign = SIGN[curand_uniform(&localState) < 0.5f];
                const float order_deviation = (curand_uniform(&localState) - 0.5f) * 5;
                sharedPopulation[tid + THREADS_PER_BLOCK][i] += powf(10.0f, order + order_deviation) * sign * mult;
            }
        }

        // sharing a part of population with others
        if ((blockIdx.x + generationIndex) % 5 == 0) {
            for (int i=0; i<VAR_NUMBER; ++i)
                population[gid * VAR_NUMBER + i] = sharedPopulation[tid][i];
        }

        // take some best individuals from neighbour
        if ((blockIdx.x + generationIndex) % 3 == 0) {
            if (curand_uniform(&localState) < 0.11) {
                const int anotherBlock = curand_uniform(&localState) * (POPULATION_SIZE / THREADS_PER_BLOCK);
                const int ngid  = blockDim.x * anotherBlock + threadIdx.x;
                for (int i=0; i<VAR_NUMBER; ++i)
                    sharedPopulation[tid][i] = population[ngid * VAR_NUMBER + i];
                sharedScore[tid] = rosenbrock(sharedPopulation[tid]);
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
    cudasafe(cudaSetDevice(0), "Could not set device 0");

	double ans = 0;

	float *population = new float[POPULATION_SIZE * VAR_NUMBER];

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i * VAR_NUMBER + u] = (float_random() - 0.5f) * 10;
		}
	}

	// copying population to device
	float *devicePopulation = 0;
	float *nextGeneration = 0;
	ScoreWithId *deviceScore = 0;
	curandState* randomStates;

	cudasafe(cudaMalloc(&randomStates, THREADS_PER_BLOCK * sizeof(curandState)), "Could not allocate memory for randomStates");
	cudasafe(cudaMalloc((void **)&devicePopulation, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for devicePopulation");
	cudasafe(cudaMalloc((void **)&nextGeneration, POPULATION_SIZE * VAR_NUMBER * sizeof(float)), "Could not allocate memory for nextGeneration");
	cudasafe(cudaMalloc((void **)&deviceScore, POPULATION_SIZE * sizeof (ScoreWithId)), "Could not allocate memory for deviceScore");

	cudasafe(cudaMemcpy(devicePopulation, population, POPULATION_SIZE * VAR_NUMBER * sizeof(float), cudaMemcpyHostToDevice), "Could not copy population to device");

	// invoking random init
	randomInit<<<1, THREADS_PER_BLOCK>>>(randomStates, 900);
	cudasafe(cudaGetLastError(), "Could not invoke kernel randomInit");
	cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling randomInit");

	const int BLOCKS_NUMBER = (POPULATION_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    //for (int i=0; i<1115; i++) {
        GAKernel<<<BLOCKS_NUMBER, THREADS_PER_BLOCK>>>(devicePopulation, deviceScore, randomStates);
        cudasafe(cudaGetLastError(), "Could not invoke GAKernel");
        cudasafe(cudaDeviceSynchronize(), "Failed to syncrhonize device after calling GAKernel");

        printPopulation(devicePopulation, deviceScore);
    //}

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
	srand(1900);
	srand(static_cast<unsigned>(time(0)));

	double ans = solveGPU();
	std::cout << "GPU answer = " << ans << std::endl;

	return 0;
}
