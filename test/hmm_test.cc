// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the HMM code.

#include "gtest/gtest.h"

#include "../hmm.h"
#include "../../core/corpus.h"
#include "../../core/util.h"

// Test class that provides a simple labeled dataset.
class LabeledDataExample : public testing::Test {
protected:
    virtual void SetUp() {
	data_file_path_ = tmpnam(nullptr);
	output_directory_path_ = tmpnam(nullptr);

	ofstream data_file(data_file_path_, ios::out);
	data_file << "the" + separater_ + "D ";
	data_file << "dog" + separater_ + "N ";
	data_file << "saw" + separater_ + "V ";
	data_file << "the" + separater_ + "D ";
	data_file << "cat" + separater_ + "N ";
	data_file << "." + separater_ + "S" << endl;

	data_file << "the" + separater_ + "D ";
	data_file << "dog" + separater_ + "N ";
	data_file << "barked" + separater_ + "V ";
	data_file << "." + separater_ + "S" << endl;

	data_file << "the" + separater_ + "D ";
	data_file << "cat" + separater_ + "N ";
	data_file << "laughed" + separater_ + "V ";
	data_file << "!" + separater_ + "S" << endl;

	// MLE parameter estimates.
	rare0_emission_["D"]["the"] = 1.0;
	rare0_emission_["N"]["dog"] = 0.5;
	rare0_emission_["N"]["cat"] = 0.5;
	rare0_emission_["V"]["saw"] = 1.0 / 3.0;
	rare0_emission_["V"]["barked"] = 1.0 / 3.0;
	rare0_emission_["V"]["laughed"] = 1.0 / 3.0;
	rare0_emission_["S"]["."] = 2.0 / 3.0;
	rare0_emission_["S"]["!"] = 1.0 / 3.0;
	transition_["D"]["N"] = 1.0;
	transition_["N"]["V"] = 0.75;
	transition_["N"]["S"] = 0.25;
	transition_["V"]["D"] = 1.0 / 3.0;
	transition_["V"]["S"] = 2.0 / 3.0;
	prior_["D"] = 1.0;
	stop_["S"] = 1.0;
    }

    virtual void TearDown() {
	remove(data_file_path_.c_str());
	ASSERT(system(("rm -rf " + output_directory_path_).c_str()) == 0,
	       "Cannot remove: " << output_directory_path_);
    }

    string data_file_path_;
    string output_directory_path_;
    const string separater_ = "__<label>__";
    double tol_ = 1e-10;
    unordered_map<string, unordered_map<string, double> > rare0_emission_;
    unordered_map<string, unordered_map<string, double> > transition_;
    unordered_map<string, double> prior_;
    unordered_map<string, double> stop_;
};

// Checks supervised training with rare cutoff 0.
TEST_F(LabeledDataExample, CheckSupervisedTrainingRare0) {
    HMM hmm;
    hmm.set_rare_cutoff(0);
    hmm.set_verbose(false);
    hmm.TrainSupervised(data_file_path_);
    for (const auto &state_pair: rare0_emission_) {
	for (const auto &observation_pair: state_pair.second) {
	    EXPECT_NEAR(observation_pair.second,
			hmm.EmissionProbability(state_pair.first,
						observation_pair.first), tol_);
	}
    }
    for (const auto &state1_pair: transition_) {
	for (const auto &state2_pair: state1_pair.second) {
	    EXPECT_NEAR(state2_pair.second,
			hmm.TransitionProbability(state1_pair.first,
						  state2_pair.first), tol_);
	}
    }
    for (const auto &state_pair: prior_) {
	EXPECT_NEAR(state_pair.second,
		    hmm.PriorProbability(state_pair.first), tol_);
    }
    for (const auto &state_pair: stop_) {
	EXPECT_NEAR(state_pair.second,
		    hmm.StoppingProbability(state_pair.first), tol_);
    }
}

// Checks supervised training with rare cutoff 1.
TEST_F(LabeledDataExample, CheckSupervisedTrainingRare1) {
    HMM hmm;
    hmm.set_rare_cutoff(1);
    hmm.set_verbose(false);
    hmm.TrainSupervised(data_file_path_);

    // V -> <?>: 1.0
    // S -> <?>: 1.0 / 3.0;
    EXPECT_NEAR(1.0, hmm.EmissionProbability("V", corpus::kRareString), tol_);
    EXPECT_NEAR(1.0 / 3.0,
		hmm.EmissionProbability("S", corpus::kRareString), tol_);
}

// Checks saving and loading a trained model
TEST_F(LabeledDataExample, CheckSavingAndLoadingTrainedModel) {
    HMM hmm1(output_directory_path_);
    hmm1.set_rare_cutoff(1);
    hmm1.set_verbose(false);
    hmm1.TrainSupervised(data_file_path_);
    hmm1.Save();

    HMM hmm2(output_directory_path_);
    hmm2.Load();
    // V -> <?>: 1.0
    // S -> <?>: 1.0 / 3.0;
    EXPECT_NEAR(1.0,
		hmm2.EmissionProbability("V", corpus::kRareString), tol_);
    EXPECT_NEAR(1.0 / 3.0,
		hmm2.EmissionProbability("S", corpus::kRareString), tol_);
}

// Test class that provides a random HMM.
class RandomHMM : public testing::Test {
protected:
    virtual void SetUp() {
	hmm_.CreateRandomly(num_observations_, num_states_);
	srand(time(NULL));
	for (size_t i = 0; i < length_; ++i) {
	    observation_string_sequence_.push_back(
		"observation" + to_string(rand() % num_observations_));
	}
    }
    HMM hmm_;
    size_t num_observations_ = 6;
    size_t num_states_ = 3;  // Do not use a large value!
    size_t length_ = 8;  // Do not use a large value!
    vector<string> observation_string_sequence_;
};

// Checks the correctness of Viterbi decoding.
TEST_F(RandomHMM, Viterbi) {
    hmm_.set_decoding_method("viterbi");
    hmm_.set_verbose(false);
    hmm_.set_debug(true);  // Debugging on.
    vector<string> state_string_sequence;
    hmm_.Predict(observation_string_sequence_, &state_string_sequence);
}

// Checks the correctness of the forward-backward algorithm.
TEST_F(RandomHMM, ForwardBackward) {
    hmm_.set_verbose(false);
    hmm_.set_debug(true);  // Debugging on.
    vector<string> state_string_sequence;
    hmm_.ComputeLogProbability(observation_string_sequence_);
}

// Test class that provides a simple unlabeled dataset.
class UnlabeledDataExample : public testing::Test {
protected:
    virtual void SetUp() {
	data_file_path_ = tmpnam(nullptr);
	ofstream data_file(data_file_path_, ios::out);
	data_file << "the dog chased the cat" << endl;
	data_file << "the cat chased the mouse" << endl;
	data_file << "the mouse chased the dog" << endl;
    }

    virtual void TearDown() { remove(data_file_path_.c_str()); }

    string data_file_path_;
    double tol_ = 1e-10;
};

// Runs unsupervised training (bw) with 0 rare cutoff, 3 hidden states.
TEST_F(UnlabeledDataExample, RunUnsupervisedTrainingBWRare0State3NoCheck) {
    HMM hmm;
    hmm.set_rare_cutoff(0);
    hmm.set_unsupervised_learning_method("bw");
    hmm.set_verbose(false);
    hmm.TrainUnsupervised(data_file_path_, 3);

    // Will not check explicitly. A few observed local optima were:
    // Likelihood: -10.75
    //    state1 state2 state0 state1 state2
    //    state1 state2 state0 state1 state2
    //    state1 state2 state0 state1 state2
    //
    // Likelihood: -19.07
    //    state2 state1 state2 state1 state0
    //    state2 state1 state2 state1 state0
    //    state2 state1 state2 state1 state0
    //
    // Likelihood: -22.21
    //    state0 state2 state0 state0 state1
    //    state0 state1 state0 state0 state2
    //    state0 state2 state0 state0 state2
    //
    // Likelihood: -23.23
    //    state2 state2 state0 state1 state1
    //    state2 state2 state0 state1 state1
    //    state2 state2 state0 state1 state1
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
