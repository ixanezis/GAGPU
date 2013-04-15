const int VAR_NUMBER = 5;
const double KNOWN_ANSWER = 0;
const int POPULATION_SIZE = 1024*6; // should be a multiple of block size
const unsigned U_RAND_MAX = static_cast<unsigned>(RAND_MAX) + 1;

// random number in [0, 1)
float float_random() {
	return (static_cast<float>(rand())) / (U_RAND_MAX);
}

struct ScoreWithId {
	float score;
	int id;
};

