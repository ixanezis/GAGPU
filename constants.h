const int VAR_NUMBER = 3;
const double KNOWN_ANSWER = 0;
const int POPULATION_SIZE = 1000;

// random number in [0, 1)
float float_random() {
	return static_cast<float>(rand()) / (RAND_MAX + 1);
}

struct ScoreWithId {
	float score;
	int id;
};

