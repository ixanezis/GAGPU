#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include "constants.h"

double random_cache[1 << 16];
int nxtRandom = 0;
double fast_float_random() {
	++nxtRandom; nxtRandom &= ((1 << 16) - 1);
	return random_cache[nxtRandom];
}

template <class T>
inline T sqr(const T& value) {
	return value * value;
}

void calcScore(const float *population, ScoreWithId* score) {
	for (int i=0; i<POPULATION_SIZE; ++i) {
		float result = 0;
		const float *curPos = &population[i * VAR_NUMBER];
		for (size_t u=0; u<VAR_NUMBER-1; ++u) {
			result += sqr(1 - *curPos) + 100 * sqr(*(curPos+1) - sqr(*curPos));
		}
		score[i].score = result;
		score[i].id = i;
	}
}

void produceGeneration(const float* population, float* nextGeneration, ScoreWithId* score) {
	for (int i=0; i<POPULATION_SIZE; ++i) {

		float* nextGenerationPos = &nextGeneration[i * VAR_NUMBER];
		const float* individual = &population[score[i % (POPULATION_SIZE / 3)].id * VAR_NUMBER];

		if (i < POPULATION_SIZE / 3) { // copy as is
			for (int i=0; i<VAR_NUMBER; ++i) {
				*nextGenerationPos = *individual;
				++nextGenerationPos;
				++individual;
			}
		} else {
			if (i < POPULATION_SIZE * 2 / 3) { // mutate
				for (int i=0; i<VAR_NUMBER; ++i) {
					*nextGenerationPos = *individual + powf(10.0, ((float_random() * 17) - 15)) * (float_random() < 0.5f ? -1 : 1);
					++nextGenerationPos;
					++individual;
				}
			} else { // crossover
				const int otherIndividualIndex = (i + static_cast<int>(float_random() * POPULATION_SIZE)) % (POPULATION_SIZE / 3);
				const float* otherIndividual = &population[score[otherIndividualIndex].id * VAR_NUMBER];

				for (int i=0; i<VAR_NUMBER; ++i) {
					*nextGenerationPos = (*individual + *otherIndividual) * 0.0f;
					++nextGenerationPos;
					++individual;
					++otherIndividual;
				}
			}
		}
	}
}

struct ScoreCompare {
	bool operator() (const ScoreWithId& a, const ScoreWithId& b) const {
		return a.score < b.score;
	}
};


double solveCPU() {
	float *population = new float[POPULATION_SIZE * VAR_NUMBER];
	float *nextGeneration = new float[POPULATION_SIZE * VAR_NUMBER];
	ScoreWithId *score = new ScoreWithId[POPULATION_SIZE];

	for (int i=0; i<POPULATION_SIZE; ++i) {
		for (int u=0; u<VAR_NUMBER; ++u) {
			population[i * VAR_NUMBER + u] = float_random();
		}
	}

	for (int generationIndex=0; generationIndex < 30000; ++generationIndex) {
		// invoking calcScore
		calcScore(population, score);

		std::sort(score, score + POPULATION_SIZE, ScoreCompare());

		produceGeneration(population, nextGeneration, score);

		std::swap(population, nextGeneration);

		std::cout << "printing first 10 elements of score:" << std::endl;
		for (int i=0; i<10; i++)
			std::cout << score[i].score << ' ';
		std::cout << std::endl;

		
		if (fabs(score[0].score - KNOWN_ANSWER) < 1e-12) {
			std::cout << "result found on generation " << generationIndex << std::endl;
			break;
		}
	}

	delete[] population;
	delete[] nextGeneration;
	delete[] score;

	return score[0].score;
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	for (int i=0; i<(1 << 16); ++i) {
		random_cache[i] = float_random();
	}

	clock_t start = clock();
	double ans = solveCPU();
	std::cout << "CPU answer = " << ans << std::endl;
	std::cout << "Time taken on CPU = " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

	return 0;
}