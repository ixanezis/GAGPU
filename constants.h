const int VAR_NUMBER = 4;
const double KNOWN_ANSWER = 0;
const int POPULATION_SIZE = 1000;
const unsigned U_RAND_MAX = static_cast<unsigned>(RAND_MAX) + 1;

// random number in [0, 1)
float float_random() {
	return (static_cast<float>(rand())) / (U_RAND_MAX);
}

struct ScoreWithId {
	float score;
	int id;
};

