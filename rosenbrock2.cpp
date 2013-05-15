#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <cstring>
#include "constants.h"

const float PI_FLOAT = acos(-1.0f);

float random_cache[1 << 16];
int nxtRandom = 0;
float fast_float_random() {
	++nxtRandom; nxtRandom &= ((1 << 16) - 1);
	return random_cache[nxtRandom];
}

template <class T>
inline T sqr(const T& value) {
	return value * value;
}

float rosenbrock(const float* curPos) {
    float result = 0;
    for (size_t i=0; i<VAR_NUMBER-1; ++i) {
        result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
        ++curPos;
    }
    return result;
}

float rastrigin(const float *curPos) {
    float result = 10.0f * VAR_NUMBER;
    for (size_t i=0; i<VAR_NUMBER; ++i) {
        result += *curPos * *curPos - 10.0f * cosf(2 * PI_FLOAT * *curPos);
        ++curPos;
    }
    return result;
}

double solveCPU() {
	float *population = new float[POPULATION_SIZE * VAR_NUMBER];
    float *score = new float[POPULATION_SIZE];
	const float signs[2] = {-1.0f, 1.0f};

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i * VAR_NUMBER + u] = float_random();
		}
	}

    for (int i=0; i<POPULATION_SIZE / 2; ++i) {
        const float *curpos = population + i * VAR_NUMBER;
        score[i] = rosenbrock(curpos);
    }

	for (int generationIndex=0; ; ++generationIndex) {
		// calculating score for the second half of individuals
        for (int i=POPULATION_SIZE / 2; i<POPULATION_SIZE; ++i) {
            const float *curpos = population + i * VAR_NUMBER;
            score[i] = rosenbrock(curpos);
        }

        if (generationIndex == 400000) break;

		// selection
        // first half of individuals might be replaced with corresponding from another half
        for (int i=0; i<POPULATION_SIZE / 2; ++i) {
            if (score[i] > score[i + POPULATION_SIZE / 2]) {
                score[i] = score[i + POPULATION_SIZE / 2];
                memcpy(population + i * VAR_NUMBER, population + (i + POPULATION_SIZE / 2) * VAR_NUMBER, VAR_NUMBER * sizeof(float));
            }
        }


		// now we've got best individuals in the first half of sharedPopulation
        for (int i=POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
            int first = rand() % (POPULATION_SIZE / 2);
            int second = rand() % (POPULATION_SIZE / 2);
            
            const float weight = fast_float_random();
			for (int u=0; u<VAR_NUMBER; ++u) {
				population[i * VAR_NUMBER + u] = population[first * VAR_NUMBER + u] * weight + population[second * VAR_NUMBER + u] * (1.0f - weight);
			}
        }
            
		// mutations on second half of population
        for (int i=POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
            if (fast_float_random() < 0.8) {
                const float order = (fast_float_random() * 17) - 15;
                for (int u=0; u<VAR_NUMBER; ++u) {
                    if (fast_float_random() < 0.8) {
                        const float sign = signs[rand() % 2];
                        const float order_deviation = (fast_float_random() - 0.5f) * 5;
                        population[i * VAR_NUMBER + u] += powf(10.0, order + order_deviation) * sign;
                    }
                }
            }
        }

		if (generationIndex % 1000 == 0) {
			std::cout << "generationIndex = " << generationIndex << std::endl
                      << "printing first 10 elements of score:" << std::endl;

			for (int i=0; i<10; i++)
				std::cout << score[i] << ' ';
			std::cout << std::endl;
		}
		
        /*
		if (fabs(score[0].score - KNOWN_ANSWER) < 1e-12) {
			std::cout << "result found on generation " << generationIndex << std::endl;
			break;
		}
        */
	}

	std::cout << "Best solution: " << std::endl;
	for (int i=0; i<VAR_NUMBER; ++i) {
		std::cout << population[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << "score = " << score[0] << std::endl;

	delete[] population;
	delete[] score;

	return score[0];
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	for (int i=0; i<(1 << 16); ++i) {
		random_cache[i] = float_random();
	}

	//freopen("output-cpu.txt", "w", stdout);
	clock_t start = clock();
	double ans = solveCPU();
	//std::cout << "CPU answer = " << ans << std::endl;
	std::cout << "Time taken on CPU = " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

	return 0;
}
