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

const double multiples[] = {1000, 100, 10, 0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13};
const int numMultiples = sizeof (multiples) / sizeof (*multiples);


// Inididuals looking for a minimum of a Rosenbrock's function
template<class T>
struct Individual {
	Individual() {
		for (int i=0; i<VAR_NUMBER; i++) {
			coeffs.push_back(fast_float_random());
		}
	}

	void mutate() {
		T multiple = multiples[static_cast<int>(fast_float_random() * numMultiples)];

		for (size_t i=0; i<coeffs.size(); i++) {
			if (fast_float_random() < 0.3) {
				coeffs[i] += (fast_float_random() - 0.5) * multiple;
			}
		}
	}

	T calcResult() {
		//return result = sqr(1 - coeffs[0]) + 100 * sqr(coeffs[1] - sqr(coeffs[0])); // 2D case
		result = 0;
		for (size_t i=0; i<coeffs.size()-1; ++i) {
			result += sqr(1 - coeffs[i]) + 100 * sqr(coeffs[i+1] - sqr(coeffs[i]));
		}
		return result;
	}

	bool operator < (const Individual& rhs) const {
		return result < rhs.result;
	}

	friend std::ostream& operator << (std::ostream& out, const Individual& ind) {
		for (const double coeff : ind.coeffs) {
			out << ' ' << coeff;
		}
		out << std::endl << ind.result << std::endl;
		return out;
	}

	std::vector<T> coeffs;
	T result;
};

template <class T>
Individual<T> operator | (const Individual<T>& a, const Individual<T>& b) {
	Individual<T> ret;

	for (size_t i=0; i<a.coeffs.size(); i++) {
		ret.coeffs[i] = ((a.coeffs[i] + b.coeffs[i]) * 0.5);
	}

	return ret;
}

double solveCPU() {
	for (int i=0; i<(1 << 16); ++i) {
		random_cache[i] = float_random();
	}

	std::vector<Individual<double>> population(POPULATION_SIZE);

	Individual<double> best;

	for (int generationIndex = 0; generationIndex <= 30000; ++generationIndex) {
		for (auto& individual : population) {
			//std::cerr << "parsing individual = " << individual << std::endl;
			individual.calcResult();
		}

		std::sort(population.begin(), population.end());
		best = population[0];
		//std::cout << "Best for now:" << std::endl;
		//std::cout << best << std::endl;

		population.erase(population.begin() + 1000, population.end());

		const std::vector<Individual<double>> unchanged(population.begin(), population.begin() + 50);

		for (auto& individual : population) {
			if (fast_float_random() < 0.9) {
				individual.mutate();
			}
		}

		population.insert(population.end(), unchanged.begin(), unchanged.end());

		for (int i=0; i<POPULATION_SIZE / 2; i++) {
			int a = rand() % population.size();
			int b = rand() % population.size();

			population.push_back(population[a] | population[b]);
		}

		if (fabs(best.result - KNOWN_ANSWER) < 1e-12) {
			std::cout << "result found on generation " << generationIndex << std::endl;
			break;
		}
	}

	return best.result;
}

int main() {
	srand(900);
	srand(static_cast<unsigned>(time(0)));

	clock_t start = clock();
	double ans = solveCPU();
	std::cout << "CPU answer = " << ans << std::endl;
	std::cout << "Time taken on CPU = " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;

	return 0;
}