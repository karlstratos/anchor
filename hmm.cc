// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "hmm.h"

#include <iomanip>
#include <limits>
#include <numeric>
#include <random>

#include "core/eigen_helper.h"
#include "core/evaluate.h"
#include "core/features.h"
#include "core/optimize.h"
#include "core/sparsesvd.h"
#include "core/util.h"

void HMM::SetOutputDirectory(const string &output_directory) {
    ASSERT(!output_directory.empty(), "Empty output directory.");
    output_directory_ = output_directory;

    // Remove a file at the path (if it exists).
    if (util_file::exists(output_directory_) &&
	util_file::get_file_type(output_directory_) == "file") {
	ASSERT(system(("rm -f " + output_directory_).c_str()) == 0,
	       "Cannot remove file: " << output_directory_);
    }

    // Create the current output directory (if necessary).
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);
}

void HMM::ResetOutputDirectory() {
    ASSERT(!output_directory_.empty(), "No output directory given.");
    ASSERT(system(("rm -f " + output_directory_ + "/*").c_str()) == 0,
	   "Cannot remove the content in: " << output_directory_);
    SetOutputDirectory(output_directory_);
}

void HMM::Clear() {
    observation_dictionary_.clear();
    observation_dictionary_inverse_.clear();
    state_dictionary_.clear();
    state_dictionary_inverse_.clear();
    emission_.clear();
    transition_.clear();
    prior_.clear();
}

void HMM::CreateRandomly(size_t num_observations, size_t num_states) {
    Clear();

    // Create an observation dictionary.
    for (Observation observation = 0; observation < num_observations;
	 ++observation) {
	string observation_string = "observation" + to_string(observation);
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }

    // Create a state dictionary.
    for (State state = 0; state < num_states; ++state) {
	string state_string = "state" + to_string(state);
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }

    InitializeParametersRandomly();
}

void HMM::Save(const string &model_path) {
    ofstream model_file(model_path, ios::out | ios::binary);
    util_file::binary_write_primitive(lowercase_, model_file);
    util_file::binary_write_primitive(rare_cutoff_, model_file);
    size_t num_observations = NumObservations();
    size_t num_states = NumStates();
    util_file::binary_write_primitive(num_observations, model_file);
    util_file::binary_write_primitive(num_states, model_file);
    for (const auto &observation_pair : observation_dictionary_) {
	string observation_string = observation_pair.first;
	Observation observation = observation_pair.second;
	util_file::binary_write_string(observation_string, model_file);
	util_file::binary_write_primitive(observation, model_file);
    }
    for (const auto &state_pair : state_dictionary_) {
	string state_string = state_pair.first;
	State state = state_pair.second;
	util_file::binary_write_string(state_string, model_file);
	util_file::binary_write_primitive(state, model_file);
    }
    for (State state = 0; state < emission_.size(); ++state) {
	for (Observation observation = 0; observation < emission_[state].size();
	     ++observation) {
	    double value = emission_[state][observation];
	    util_file::binary_write_primitive(value, model_file);
	}
    }
    for (size_t state1 = 0; state1 < transition_.size(); ++state1) {
	for (size_t state2 = 0; state2 < transition_[state1].size(); ++state2) {
	    double value = transition_[state1][state2];
	    util_file::binary_write_primitive(value, model_file);
	}
    }
    for (size_t state = 0; state < prior_.size(); ++state) {
	double value = prior_[state];
	util_file::binary_write_primitive(value, model_file);
    }
}

void HMM::Load(const string &model_path) {
    Clear();
    ifstream model_file(model_path, ios::in | ios::binary);
    size_t num_observations;
    size_t num_states;
    util_file::binary_read_primitive(model_file, &lowercase_);
    util_file::binary_read_primitive(model_file, &rare_cutoff_);
    util_file::binary_read_primitive(model_file, &num_observations);
    util_file::binary_read_primitive(model_file, &num_states);
    for (size_t i = 0; i < num_observations; ++i) {
	string observation_string;
	Observation observation;
	util_file::binary_read_string(model_file, &observation_string);
	util_file::binary_read_primitive(model_file, &observation);
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }
    for (size_t i = 0; i < num_states; ++i) {
	string state_string;
	State state;
	util_file::binary_read_string(model_file, &state_string);
	util_file::binary_read_primitive(model_file, &state);
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }
    emission_.resize(num_states);
    for (State state = 0; state < num_states; ++state) {
	emission_[state].resize(num_observations,
				-numeric_limits<double>::infinity());
	for (Observation observation = 0; observation < num_observations;
	     ++observation) {
	    double value;
	    util_file::binary_read_primitive(model_file, &value);
	    emission_[state][observation] = value;
	}
    }
    transition_.resize(num_states);
    for (State state1 = 0; state1 < num_states; ++state1) {
	transition_[state1].resize(num_states + 1,  // +stop
				   -numeric_limits<double>::infinity());
	for (State state2 = 0; state2 < num_states + 1; ++state2) {  // +stop
	    double value;
	    util_file::binary_read_primitive(model_file, &value);
	    transition_[state1][state2] = value;
	}
    }
    prior_.resize(num_states, -numeric_limits<double>::infinity());
    for (State state = 0; state < num_states; ++state) {
	double value;
	util_file::binary_read_primitive(model_file, &value);
	prior_[state] = value;
    }
    CheckProperDistribution();
}

void HMM::WriteModelInfo(const string &info_path) {
    // Get max string legnths for pretty outputs.
    size_t max_state_string_length = 0;
    size_t max_observation_string_length = 0;
    for (State state = 0; state < NumStates(); ++state) {
	string state_string = state_dictionary_inverse_[state];
	if (state_string.size() > max_state_string_length) {
	    max_state_string_length = state_string.size();
	}
    }
    for (Observation observation = 0; observation < NumObservations();
	 ++observation) {
	string observation_string =
	    observation_dictionary_inverse_[observation];
	if (observation_string.size() > max_observation_string_length) {
	    max_observation_string_length = observation_string.size();
	}
    }
    const size_t buffer = 3;

    // Write string representations of hidden states.
    ofstream info_file(info_path, ios::out);
    info_file << endl << "STATE STRINGS" << endl;
    for (State state = 0; state < NumStates(); ++state) {
	string state_string = state_dictionary_inverse_[state];
	string line = util_string::printf_format(
	    "State %d \"%s\"", state + 1,
	    state_dictionary_inverse_[state].c_str());
	info_file << line << endl;
    }

    // For each state, write most likely observations.
    info_file << endl << "TOP EMISSION PROBABILITIES" << endl;
    for (State state = 0; state < NumStates(); ++state) {
	vector<pair<string, double> > v;
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    v.emplace_back(observation_dictionary_inverse_[observation],
			   exp(emission_[state][observation]));
	}
	sort(v.begin(), v.end(),
	     util_misc::sort_pairs_second<string, double,
	     greater<double> >());

	string line = util_string::buffer_string(
	    util_string::printf_format(
		"\n%s", state_dictionary_inverse_[state].c_str()),
	    max_state_string_length + buffer, ' ', "left");
	info_file << line;
	for (size_t i = 0; i < min((int) v.size(), 20); ++i) {  // Top 20
	    line = util_string::printf_format("\n%s (%.2f)", v[i].first.c_str(),
					      v[i].second);
	    info_file << line;
	}
	info_file << endl;
    }

    // Write prior probabilities for states.
    info_file << endl << "PRIOR PROBABILITIES" << endl;
    vector<pair<string, double> > v;
    for (State state = 0; state < NumStates(); ++state) {
	v.emplace_back(state_dictionary_inverse_[state],
		       exp(prior_[state]));
    }
    sort(v.begin(), v.end(), util_misc::sort_pairs_second<string,
	 double, greater<double> >());
    for (State state = 0; state < NumStates(); ++state) {
	string state_string = util_string::buffer_string(
	    v[state].first, max_state_string_length + buffer, ' ', "right");
	string line = util_string::printf_format("%s   %.2f",
						 state_string.c_str(),
						 v[state].second);
	info_file << line << endl;
    }

    // Write transition  probabilities between states.
    info_file << endl << "TOP TRANSITION PROBABILITIES" << endl;
    for (State state = 0; state < NumStates(); ++state) {
	vector<pair<string, double> > v;
	for (State next_state = 0; next_state < NumStates(); ++next_state) {
	    v.emplace_back(state_dictionary_inverse_[next_state],
			   exp(transition_[state][next_state]));
	}
	sort(v.begin(), v.end(), util_misc::sort_pairs_second<string,
	     double, greater<double> >());

	string line = util_string::buffer_string(
	    util_string::printf_format(
		"%s", state_dictionary_inverse_[state].c_str()),
	    max_state_string_length + buffer, ' ', "left");
	info_file << line;
	for (size_t i = 0; i < min((int) v.size(), 5); ++i) {  // Top 5
	    line = util_string::buffer_string(
		util_string::printf_format("%s (%.2f) ",
					   v[i].first.c_str(),
					   v[i].second),
		max_state_string_length + buffer + 7, ' ', "right");
	    info_file << line;
	}
	info_file << endl;
    }
}

void HMM::TrainSupervised(const string &data_path) {
    Clear();

    // First, construct observation and state dictionaries.
    ConstructDictionaries(data_path, true);

    // Next, collect co-occurrence counts...
    vector<vector<size_t> > emission_count(NumObservations());
    for (State state = 0; state < NumStates(); ++state) {
	emission_count[state].resize(NumObservations(), 0);
    }
    vector<vector<size_t> > transition_count(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	transition_count[state].resize(NumStates() + 1, 0);  // +stop
    }
    vector<size_t> prior_count(NumStates(), 0);

    // ... from the labeled data.
    ifstream data_file(data_path, ios::in);
    ASSERT(data_file.is_open(), "Cannot open " << data_path);
    vector<string> observation_string_sequence;
    vector<string> state_string_sequence;
    while (ReadLine(true, &data_file, &observation_string_sequence,
		    &state_string_sequence)) {
	size_t length = observation_string_sequence.size();
	vector<Observation> observation_sequence;
	vector<State> state_sequence;
	ConvertObservationSequence(observation_string_sequence,
				   &observation_sequence);
	ConvertStateSequence(state_string_sequence, &state_sequence);

	++prior_count[state_sequence[0]];  // Initial State
	for (size_t i = 0; i < length; ++i) {
	    Observation observation = observation_sequence[i];
	    State state = state_sequence[i];
	    ++emission_count[state][observation];  // State -> Observation
	    if (i > 0) {
		State previous_state = state_sequence[i - 1];
		++transition_count[previous_state][state];  // State' -> State
	    }
	}
	State final_state = state_sequence[length - 1];
	++transition_count[final_state][StoppingState()];  // State -> <STOP>
    }

    // Finally, set maximum-likelihood parameter values.
    emission_.resize(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	size_t state_normalizer = accumulate(emission_count[state].begin(),
					     emission_count[state].end(), 0);
	emission_[state].resize(NumObservations(),
				-numeric_limits<double>::infinity());
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    emission_[state][observation] =
		log(emission_count[state][observation]) - log(state_normalizer);
	}
    }
    transition_.resize(NumStates());
    for (State state1 = 0; state1 < NumStates(); ++state1) {
	size_t state1_normalizer = accumulate(transition_count[state1].begin(),
					      transition_count[state1].end(),
					      0);
	transition_[state1].resize(NumStates() + 1,  // +stop
				   -numeric_limits<double>::infinity());
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    transition_[state1][state2] =
		log(transition_count[state1][state2]) - log(state1_normalizer);
	}
    }
    size_t prior_normalizer =
	accumulate(prior_count.begin(), prior_count.end(), 0);
    prior_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = log(prior_count[state]) - log(prior_normalizer);
    }
    CheckProperDistribution();

    // Report performance and likelihood on the training data.
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    ReadLines(data_path, true, &observation_string_sequences,
	      &state_string_sequences);

    // Make predictions.
    vector<vector<string> > predictions;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	vector<string> prediction;
	Predict(observation_string_sequences[i], &prediction);
	predictions.push_back(prediction);
    }
    unordered_map<string, string> label_mapping;
    double position_accuracy;
    double sequence_accuracy;
    eval_sequential::compute_accuracy_mapping_labels(
	state_string_sequences, predictions, &position_accuracy,
	&sequence_accuracy, &label_mapping);
    double likelihood = ComputeLogProbability(observation_string_sequences);
    Report(util_string::printf_format(
	       "\n---TRAINING---\n"
	       "File: %s\n"
	       "%s (many-to-one)   "
	       "per-position: %.2f%%   "
	       "per-sequence: %.2f%%   "
	       "likelihood: %.2f",
	       util_file::get_file_name(data_path).c_str(),
	       decoding_method_.c_str(), position_accuracy,
	       sequence_accuracy, likelihood));
}

void HMM::TrainUnsupervised(const string &data_path, size_t num_states) {
    Clear();

    // First, construct observation dictionaries from the data.
    unordered_map<Observation, size_t> observation_count;
    ConstructDictionaries(data_path, false, &observation_count);

    // Also, construct state dictionaries (already clear) synthetically.
    for (State state = 0; state < num_states; ++state) {
	AddStateIfUnknown(kState_ + to_string(state));
    }

    // Count observation transitions for count-based methods below.
    Corpus corpus(data_path, verbose_);
    corpus.set_lowercase(lowercase_);
    unordered_map<Observation,
		  unordered_map<Observation, size_t> > observation_bigram_count;
    unordered_map<Observation, size_t> initial_observation_count;
    unordered_map<Observation, size_t> final_observation_count;
    corpus.CountTransitions(observation_dictionary_,
			    &observation_bigram_count,
			    &initial_observation_count,
			    &final_observation_count);

    // Use one of the unsupervised learning methods.
    if (unsupervised_learning_method_ == "cluster") {  // (Count-based)
	// Set parameters using clusters.
	InitializeParametersFromClusters(observation_bigram_count,
					 initial_observation_count,
					 final_observation_count);
    } else if (unsupervised_learning_method_ == "bw") {  // (Iterative)
	// Run Baum-Welch on the data.
	RunBaumWelch(data_path);
    } else if (unsupervised_learning_method_ == "anchor") {  // (Count-based)
	// Build a convex hull of observation vectors (rows of a matrix).
	Eigen::MatrixXd convex_hull;
	BuildConvexHull(&corpus, &convex_hull);

	// Compute the "flipped" emission distributions p(State|Observation)
	// (rows of a matrix).
	Eigen::MatrixXd flipped_emission;
	ComputeFlippedEmission(convex_hull, observation_count,
			       &flipped_emission);

	// Finally, recover the model parameters from the flipped emission.
	RecoverParametersGivenFlippedEmission(flipped_emission,
					      observation_count,
					      observation_bigram_count,
					      initial_observation_count,
					      final_observation_count);
    } else {
	ASSERT(false, "Unknown unsupervised learning method: "
	       << unsupervised_learning_method_);
    }

    // (Optional) Perform post-training local search.
    if (post_training_local_search_ && unsupervised_learning_method_ != "bw") {
	RunBaumWelch(data_path);
    }

    // Report likelihood on the training data.
    double likelihood = 0.0;
    ifstream data_file(data_path, ios::in);
    ASSERT(data_file.is_open(), "Cannot open " << data_path);
    vector<string> observation_string_sequence;
    vector<string> state_string_sequence;  // Unused.
    while (ReadLine(false, &data_file, &observation_string_sequence,
		    &state_string_sequence)) {
	likelihood += ComputeLogProbability(observation_string_sequence);
    }
    Report(util_string::printf_format(
	       "\n---TRAINING---\n"
	       "File: %s\n"
	       "likelihood: %.2f",
	       util_file::get_file_name(data_path).c_str(), likelihood));
}

void HMM::Evaluate(const string &labeled_data_path,
		   const string &prediction_path) {
    // Load the test data.
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    ReadLines(labeled_data_path, true, &observation_string_sequences,
	      &state_string_sequences);

    // Make predictions.
    vector<vector<string> > predictions;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	vector<string> prediction;
	Predict(observation_string_sequences[i], &prediction);
	predictions.push_back(prediction);
    }

    // For simplicity, always report the many-to-one accuracy.
    unordered_map<string, string> label_mapping;
    double position_accuracy;
    double sequence_accuracy;
    eval_sequential::compute_accuracy_mapping_labels(
	state_string_sequences, predictions, &position_accuracy,
	&sequence_accuracy, &label_mapping);
    double likelihood = ComputeLogProbability(observation_string_sequences);
    if (verbose_) {
	string line = util_string::printf_format(
	    "\n---EVALUATION---\n"
	    "%s (many-to-one) per-position: %.2f%%   per-sequence: %.2f%%"
	    "   likelihood: %.2f",
	    decoding_method_.c_str(), position_accuracy, sequence_accuracy,
	    likelihood);
	cerr << line << endl;
    }

    // If the prediction path is not "", write predictions in that file.
    if (!prediction_path.empty()) {
	ofstream file(prediction_path, ios::out);
	ASSERT(file.is_open(), "Cannot open file: " << prediction_path);
	for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	    size_t length = observation_string_sequences[i].size();
	    for (size_t j = 0; j < length; ++j) {
		file << observation_string_sequences[i][j] << " ";
		file << state_string_sequences[i][j] << " ";
		file << label_mapping[predictions[i][j]] << " ";
		file << predictions[i][j] << endl;
	    }
	    file << endl;
	}
    }
}

void HMM::Predict(const vector<string> &observation_string_sequence,
		  vector<string> *state_string_sequence) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    vector<State> state_sequence;
    if (decoding_method_ == "viterbi") {
	Viterbi(observation_sequence, &state_sequence);
    } else if (decoding_method_ == "mbr") {
	MinimumBayesRisk(observation_sequence, &state_sequence);
    } else if (decoding_method_ == "greedy") {
	GreedyDecoding(observation_sequence, &state_sequence);
    } else {
	ASSERT(false, "Unknown decoding method: " << decoding_method_);
    }
    ConvertStateSequence(state_sequence, state_string_sequence);
}

double HMM::ComputeLogProbability(
    const vector<string> &observation_string_sequence) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    return ComputeLogProbability(observation_sequence);
}

double HMM::ComputeLogProbability(
    const vector<vector<string> > &observation_string_sequences) {
    double log_probability = 0.0;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	log_probability +=
	    ComputeLogProbability(observation_string_sequences.at(i));
    }
    return log_probability;
}

void HMM::ComputeLogMarginal(const vector<string> &observation_string_sequence,
			     vector<vector<double> > *marginal) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    ComputeLogMarginal(observation_sequence, marginal);
}

void HMM::ReadLines(const string &file_path, bool labeled,
		    vector<vector<string> > *observation_string_sequences,
		    vector<vector<string> > *state_string_sequences) {
    observation_string_sequences->clear();
    state_string_sequences->clear();
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    vector<string> observation_string_sequence;
    vector<string> state_string_sequence;
    while (ReadLine(labeled, &file, &observation_string_sequence,
		    &state_string_sequence)) {
	observation_string_sequences->push_back(observation_string_sequence);
	state_string_sequences->push_back(state_string_sequence);
    }
}

bool HMM::ReadLine(bool labeled, ifstream *file,
		   vector<string> *observation_string_sequence,
		   vector<string> *state_string_sequence) {
    observation_string_sequence->clear();
    state_string_sequence->clear();
    while (file->good()) {
	vector<string> tokens;
	util_file::read_line(file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	for (size_t i = 0; i < tokens.size(); ++i) {
	    if (labeled) {
		vector<string> seperated_token;
		util_string::split_by_string(tokens[i],
					     kObservationStateSeperator_,
					     &seperated_token);
		ASSERT(seperated_token.size() == 2, "Wrong format for labeled "
		       "data with seperator \"" << kObservationStateSeperator_
		       << "\": " << tokens[i]);
		observation_string_sequence->push_back(seperated_token[0]);
		state_string_sequence->push_back(seperated_token[1]);
	    } else {
		observation_string_sequence->push_back(tokens[i]);
		state_string_sequence->push_back("");  // To match lengths.
	    }
	}
	return true;  // Successfully read a non-empty line.
    }
    return false;  // There was no more non-empty line to read.
}

double HMM::EmissionProbability(string state_string,
				string observation_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	if (observation_dictionary_.find(observation_string) !=
	    observation_dictionary_.end()) {
	    Observation observation =
		observation_dictionary_[observation_string];
	    return exp(emission_[state][observation]);
	}
    }
    return 0.0;
}

double HMM::TransitionProbability(string state1_string, string state2_string) {
    if (state_dictionary_.find(state1_string) != state_dictionary_.end()) {
	State state1 = state_dictionary_[state1_string];
	if (state_dictionary_.find(state2_string) != state_dictionary_.end()) {
	    State state2 = state_dictionary_[state2_string];
	    return exp(transition_[state1][state2]);
	}
    }
    return 0.0;
}

double HMM::PriorProbability(string state_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	return exp(prior_[state]);
    }
    return 0.0;
}

double HMM::StoppingProbability(string state_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	return exp(transition_[state][StoppingState()]);
    }
    return 0.0;
}

Observation HMM::GetObservationIndex(string observation_string) {
    if (observation_dictionary_.find(observation_string) !=
	observation_dictionary_.end()) {
	return observation_dictionary_[observation_string];
    } else {
	ASSERT(false, "Unknown observation string: " << observation_string);
    }
}

string HMM::GetObservationString(Observation observation) {
    if (observation_dictionary_inverse_.find(observation) !=
	observation_dictionary_inverse_.end()) {
	return observation_dictionary_inverse_[observation];
    } else {
	ASSERT(false, "Unknown observation: " << observation);
    }
}

State HMM::GetStateIndex(string state_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	return state_dictionary_[state_string];
    } else {
	ASSERT(false, "Unknown state string: " << state_string);
    }
}

string HMM::GetStateString(State state) {
    if (state_dictionary_inverse_.find(state) !=
	state_dictionary_inverse_.end()) {
	return state_dictionary_inverse_[state];
    } else {
	ASSERT(false, "Unknown state: " << state);
    }
}

void HMM::BuildConvexHull(Corpus *corpus, Eigen::MatrixXd *convex_hull) {
    // Extract observation-context co-occurrence counts using the observation
    // dictionary.
    unordered_map<string, Context> context_dictionary;
    unordered_map<Context, unordered_map<Observation, double> >
	context_observation_count;
    corpus->SlideWindow(observation_dictionary_, true, context_definition_,
		       window_size_, 0, &context_dictionary,
		       &context_observation_count);

    // Extend the context space (i.e., add additional columns to the
    // observation-context co-occurrence matrix).
    ExtendContextSpace(&context_dictionary, &context_observation_count);

    Report(util_string::printf_format("%d x %d word-context matrix",
				      observation_dictionary_.size(),
				      context_dictionary.size()));

    // For methods requiring CCA, pre-compute CCA projections.
    Eigen::MatrixXd cca_left_singular_vectors;
    Eigen::MatrixXd cca_right_singular_vectors;
    Eigen::VectorXd cca_singular_values;
    if (convex_hull_method_ == "brown" || convex_hull_method_ == "cca") {
	SMat cooccurrence_count_matrix =
	    sparsesvd::convert_column_map(context_observation_count);
	corpus::decompose(cooccurrence_count_matrix, NumStates(), "power",
			  add_smooth_, power_smooth_, "cca",
			  &cca_left_singular_vectors,
			  &cca_right_singular_vectors, &cca_singular_values);
	svdFreeSMat(cooccurrence_count_matrix);
    }

    if (convex_hull_method_ == "brown") {
	// Under the Brown assumption, the normalized rows of the left CCA
	// projection matrix form a (trivial) convex hull.
	(*convex_hull) = cca_left_singular_vectors;
	for (size_t i = 0; i < convex_hull->rows(); ++i) {
	    (*convex_hull).row(i).normalize();
	}
    } else if (convex_hull_method_ == "svd" || convex_hull_method_ == "cca" ||
	       convex_hull_method_ == "rand") {
	// Under the anchor assumption, the rows v(x) form a convex hull,
	// where x is an observation type corresponding to a row and
	// v_i(x) is the probability of the i-th context given x.
	vector<size_t> observation_count(observation_dictionary_.size());
	vector<size_t> context_count(context_dictionary.size());  // For CCA
	for (const auto &context_pair : context_observation_count) {
	    Context context = context_pair.first;
	    for (const auto &observation_pair : context_pair.second) {
		Observation observation = observation_pair.first;
		size_t cooccurrence_count = observation_pair.second;
		observation_count[observation] += cooccurrence_count;
		context_count[context] += cooccurrence_count;
	    }
	}
	for (const auto &context_pair : context_observation_count) {
	    Context context = context_pair.first;
	    for (const auto &observation_pair : context_pair.second) {
		Observation observation = observation_pair.first;
		context_observation_count[context][observation] /=
		    (double) observation_count[observation];
	    }
	}

	// Choose a projection from the original convex hull dimension to a
	// subspace of dimension NumStates().
	Eigen::MatrixXd projection_matrix;
	if (convex_hull_method_ == "svd") {
	    // Use the best-fit subspace projection given by the right singular
	    // vectors.
	    SMat conditional_probability_matrix =
		sparsesvd::convert_column_map(context_observation_count);
	    Eigen::MatrixXd left_singular_vectors;
	    Eigen::VectorXd singular_values;
	    size_t svd_rank;
	    sparsesvd::compute_svd(conditional_probability_matrix, NumStates(),
				   &left_singular_vectors,
				   &projection_matrix, &singular_values,
				   &svd_rank);
	    svdFreeSMat(conditional_probability_matrix);
	} else if (convex_hull_method_ == "cca") {
	    // Use the context-side CCA projection.
	    projection_matrix = cca_right_singular_vectors;
	    for (Context i = 0; i < projection_matrix.rows(); ++i) {
		projection_matrix.row(i) /= sqrt(context_count[i] +
						 add_smooth_);
	    }
	} else if (convex_hull_method_ == "rand") {
	    // Use a random projection.
	    eigen_helper::generate_random_projection(context_dictionary.size(),
						     NumStates(),
						     &projection_matrix);
	} else {
	    ASSERT(false, "This should not be reached");
	}

	// Project the original high-dimensional convex hull.
	Eigen::SparseMatrix<double, Eigen::RowMajor> original_convex_hull;
	eigen_helper::convert_column_map(context_observation_count,
					 &original_convex_hull);
	(*convex_hull) = original_convex_hull * projection_matrix;
    } else {
	ASSERT(false, "Unknown convex hull method: " << convex_hull_method_);
    }
}

void HMM::ExtendContextSpace(unordered_map<string, Context> *context_dictionary,
			     unordered_map<Context, unordered_map<Observation,
			     double> > *context_observation_count) {
    // What context extensions are specified?
    vector<string> extension_types;
    util_string::split_by_chars(context_extension_, ",", &extension_types);
    if (extension_types.size() == 0) { return; }  // No extension.

    // Go through the co-occurrence matrix to aggregate squared observation
    // counts: these are used to scale the l2 norm of the extended feature.
    vector<double> observation_squared_l2_norm(observation_dictionary_.size());
    for (const auto &context_pair : *context_observation_count) {
	for (const auto &observation_pair : context_pair.second) {
	    observation_squared_l2_norm[observation_pair.first] +=
		pow(observation_pair.second, 2);
	}
    }

    // Go through the observation strings to extend the context dictionary.
    unordered_map<Observation, vector<Context> > new_contexts;
    for (const auto &observation_string_pair : observation_dictionary_) {
	string observation_string = observation_string_pair.first;
	Observation observation = observation_string_pair.second;
	for (const string &extension_type : extension_types) {
	    if (extension_type == "basic") {
		string shape =
		    "<shape>=" + features::basic_word_shape(observation_string);
		if (context_dictionary->find(shape) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[shape] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[shape];
		new_contexts[observation].push_back(new_context);
	    } else if (extension_type == "cap") {
		string cap =
		    "<cap>=" + features::is_capitalized(observation_string);
		if (context_dictionary->find(cap) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[cap] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[cap];
		new_contexts[observation].push_back(new_context);
	    } else if (extension_type == "hyphen") {
		string hyphen =
		    "<hyphen>=" + features::contains_hyphen(observation_string);
		if (context_dictionary->find(hyphen) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[hyphen] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[hyphen];
		new_contexts[observation].push_back(new_context);
	    } else if (extension_type == "digit") {
		string digit =
		    "<digit>=" + features::contains_digit(observation_string);
		if (context_dictionary->find(digit) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[digit] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[digit];
		new_contexts[observation].push_back(new_context);
	    } else if (extension_type.substr(0, 4) == "pref") {
		string prefix_size_string = extension_type.substr(4);
		size_t prefix_size = stol(prefix_size_string);
		if (observation_string.size() < prefix_size) { continue; }
		string prefix = "<pref" + prefix_size_string + ">=" +
		    features::prefix(observation_string, prefix_size);
		if (context_dictionary->find(prefix) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[prefix] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[prefix];
		new_contexts[observation].push_back(new_context);
	    } else if (extension_type.substr(0, 4) == "suff") {
		string suffix_size_string = extension_type.substr(4);
		size_t suffix_size = stol(suffix_size_string);
		if (observation_string.size() < suffix_size) { continue; }
		string suffix = "<suff" + suffix_size_string + ">=" +
		    features::suffix(observation_string, suffix_size);
		if (context_dictionary->find(suffix) ==
		    context_dictionary->end()) {
		    (*context_dictionary)[suffix] = context_dictionary->size();
		}
		Context new_context = (*context_dictionary)[suffix];
		new_contexts[observation].push_back(new_context);
	    } else {
		ASSERT(false, "Unknown extension type: " << extension_type);
	    }
	}
    }

    for (Observation observation = 0; observation < NumObservations();
	 ++observation) {
	// For each observation row in the observation-context matrix, append a
	// new feature vector v that looks like
	//              v = (0, 0, C, 0, 0, ..., C, 0, 0)
	// The norm of v is scaled against the existing feature vector u, such
	// that:
	//             (extension_weight_) * ||u|| = ||v||
	//                                         = sqrt(v.nonzeros * C^2)
	//
	// where v.nonzeros is the number of nonzero features in v. Solve for C
	// to obtain:
	//             C = sqrt(extension_weight_^2 * ||u||^2 / v.nonzeros)
	for (Context new_context : new_contexts[observation]) {
	    (*context_observation_count)[new_context][observation] =
		sqrt(pow(extension_weight_, 2) *
		     observation_squared_l2_norm[observation] /
		     new_contexts[observation].size());
	}
    }
}

void HMM::ComputeFlippedEmission(const Eigen::MatrixXd &convex_hull,
				 const unordered_map<Observation, size_t>
				 &observation_count,
				 Eigen::MatrixXd *flipped_emission) {
    // Find anchor observations.
    vector<Observation> anchor_observations;
    FindAnchors(convex_hull, observation_count, &anchor_observations);

    state_dictionary_.clear();  // Name each state by its anchor.
    state_dictionary_inverse_.clear();
    for (State state = 0; state < anchor_observations.size(); ++state) {
	string anchor_string = observation_dictionary_inverse_[
	    anchor_observations[state]];
	state_dictionary_[anchor_string] = state;
	state_dictionary_inverse_[state] = anchor_string;
    }

    // Given anchors, compute the "flipped" emission p(State|Observation).
    optimize::extract_matrix(convex_hull, anchor_observations.size(),
			     max_num_fw_iterations_, 1e-10, verbose_,
			     anchor_observations, flipped_emission);

    // Write the flipped emission values (for the record).
    ofstream flipped_emission_file(FlippedEmissionPath(), ios::out);
    vector<pair<Observation, size_t> > sorted_observations;
    for (const auto &observation_pair : observation_count) {
	sorted_observations.emplace_back(observation_pair.first,
					 observation_pair.second);
    }
    sort(sorted_observations.begin(), sorted_observations.end(),
	 util_misc::sort_pairs_second<Observation, size_t,
	 greater<size_t> >());
    for (size_t i = 0; i < sorted_observations.size(); ++i) {
	Observation x = sorted_observations[i].first;
	vector<pair<string, double> > v;
	for (State h = 0; h < NumStates(); ++h) {
	    string anchor_string = state_dictionary_inverse_[h];
	    v.emplace_back(anchor_string, (*flipped_emission)(x, h));
	}
	sort(v.begin(), v.end(), util_misc::sort_pairs_second<string,
	     double, greater<double> >());

	flipped_emission_file << observation_dictionary_inverse_[x] << endl;
	for (State h = 0; h < NumStates(); ++h) {
	    flipped_emission_file << util_string::printf_format(
		"%.2f:   %s", v[h].second, v[h].first.c_str()) << endl;
	}
	flipped_emission_file << endl;
    }
}

void HMM::FindAnchors(const Eigen::MatrixXd &convex_hull,
		      const unordered_map<Observation, size_t>
		      &observation_count,
		      vector<Observation> *anchor_observations) {
    ASSERT(num_anchor_candidates_ >= NumStates(), "Number of anchor candidates "
	   << num_anchor_candidates_ << " < " << NumStates());

    anchor_observations->clear();
    string line = util_string::printf_format("Obtaining %d anchors",
					     NumStates());

    if (anchor_path_.empty()) {
	size_t num_candidates = (num_anchor_candidates_ >= NumStates()) ?
	    num_anchor_candidates_ : NumStates();
	line += util_string::printf_format("   (out of %d candidates)",
					   num_candidates);
	Report(line);
	unordered_map<Observation, bool> anchor_candidates;

	// Will consider only frequent observation types for anchors.
	vector<pair<Observation, size_t> > sorted_observations;
	for (const auto &observation_pair : observation_count) {
	    sorted_observations.emplace_back(observation_pair.first,
					     observation_pair.second);
	}
	sort(sorted_observations.begin(), sorted_observations.end(),
	     util_misc::sort_pairs_second<Observation, size_t,
	     greater<size_t> >());
	for (size_t i = 0; i < num_candidates; ++i) {
	    anchor_candidates[sorted_observations[i].first] = true;
	}
	optimize::find_vertex_rows(convex_hull, NumStates(), anchor_candidates,
				   anchor_observations);
    } else {
	line += util_string::printf_format("(PROVIDED BY THE USER)");
	Report(line);

	ifstream anchor_file(anchor_path_, ios::in);
	ASSERT(anchor_file.is_open(), "Cannot open " << anchor_path_);
	while (anchor_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&anchor_file, &tokens);
	    if (tokens.size() == 0) { continue; }
	    for (string token : tokens) {
		ASSERT(observation_dictionary_.find(token) !=
		       observation_dictionary_.end(), "Proposed anchor not in "
		       "the dictionary: " << token);
		Observation anchor = observation_dictionary_[token];
		if (anchor_observations->size() < NumStates()) {
		    (*anchor_observations).push_back(anchor);
		}
	    }
	}
	ASSERT(anchor_observations->size() == NumStates(), "Need "
	       << NumStates() << ", given " << anchor_observations->size());
    }

    for (size_t i = 0; i < anchor_observations->size(); ++i) {
	string anchor_string =
	    observation_dictionary_inverse_[(*anchor_observations)[i]];
	Report(util_string::printf_format("   State %d: \"%s\"", i + 1,
					  anchor_string.c_str()));
    }
}

void HMM::RecoverParametersGivenFlippedEmission(
    const Eigen::MatrixXd &flipped_emission,
    const unordered_map<Observation, size_t> &observation_count,
    const unordered_map<Observation, unordered_map<Observation, size_t> >
    &observation_bigram_count,
    const unordered_map<Observation, size_t> &initial_observation_count,
    const unordered_map<Observation, size_t> &final_observation_count) {
    // 1. Emission parameters.
    RecoverEmissionParametersGivenFlippedEmission(observation_count,
						  flipped_emission);

    // 2. Prior parameters.
    RecoverPriorParametersGivenEmission(initial_observation_count);

    // 3. Transition parameters.

    // Need to first compute average state probabilities p(h).
    Eigen::VectorXd average_observation_probabilities =
	Eigen::VectorXd::Zero(NumObservations());
    double num_observations = util_misc::sum_values(observation_count);
    for (const auto &observation_pair : observation_count) {
	average_observation_probabilities[observation_pair.first] =
	    ((double) observation_pair.second) / num_observations;
    }
    /*
    Eigen::VectorXd average_state_probabilities =  // p(h) = sum_x p(h|x) p(x)
	flipped_emission.transpose() * average_observation_probabilities;
    */
    Eigen::MatrixXd emission_matrix;
    ConstructEmissionMatrix(&emission_matrix);
    Eigen::VectorXd average_state_probabilities =  // Experimental.
	emission_matrix.transpose() * average_observation_probabilities;
    average_state_probabilities /= average_state_probabilities.lpNorm<1>();

    RecoverTransitionParametersGivenOthers(average_state_probabilities,
					   observation_bigram_count,
					   final_observation_count);
    CheckProperDistribution();
}

void HMM::RecoverEmissionParametersGivenFlippedEmission(
    const unordered_map<Observation, size_t> &observation_count,
    const Eigen::MatrixXd &flipped_emission) {
    emission_.resize(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	emission_[state].resize(NumObservations());
	double state_normalizer = 0.0;
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    // Bayes' rule: requires observation counts.
	    emission_[state][observation] =
		flipped_emission(observation, state) *
		observation_count.at(observation);
	    state_normalizer += emission_[state][observation];
	}
	// Convert to log and normalize.
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    emission_[state][observation] = log(emission_[state][observation])
		- log(state_normalizer);
	}
    }
}

void HMM::RecoverPriorParametersGivenEmission(
    const unordered_map<Observation, size_t> &initial_observation_count) {
    Eigen::VectorXd initial_observation_probabilities =
	Eigen::VectorXd::Zero(NumObservations());
    double num_initial_observations =
	util_misc::sum_values(initial_observation_count);
    for (const auto &observation_pair : initial_observation_count) {
	initial_observation_probabilities[observation_pair.first] =
	    ((double) observation_pair.second) / num_initial_observations;
    }

    Eigen::MatrixXd emission_matrix;
    ConstructEmissionMatrix(&emission_matrix);

    // Compute prior parameters as convex coefficients for emission
    // distributions resulting to become the initial observation
    // probabilities.
    /*
    Eigen::VectorXd convex_coefficients;
    optimize::compute_convex_coefficients_squared_loss(
	emission_matrix, initial_observation_probabilities,
	max_num_fw_iterations_, 1e-10, verbose_, &convex_coefficients);
    prior_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = util_math::log0(convex_coefficients[state]);
    }
    */
    Eigen::VectorXd initial_state_probabilities =  // Experimental.
	emission_matrix.transpose() * initial_observation_probabilities;
    initial_state_probabilities /= initial_state_probabilities.lpNorm<1>();
    prior_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = util_math::log0(initial_state_probabilities[state]);
    }
}

void HMM::RecoverTransitionParametersGivenOthers(
    const Eigen::VectorXd &average_state_probabilities,
    const unordered_map<Observation, unordered_map<Observation, size_t> >
    &observation_bigram_count,
    const unordered_map<Observation, size_t> &final_observation_count) {
    // Initialize transition parameters uniformly.
    InitializeTransitionParametersUniformly();

    // Keep emission and transition parameters as matrices for convenience.
    Eigen::MatrixXd emission_matrix;
    Eigen::MatrixXd transition_matrix;
    ConstructEmissionMatrix(&emission_matrix);
    ConstructTransitionMatrix(&transition_matrix);

    // Prepare data-driven model development.
    vector<vector<string> > development_observation_string_sequences;
    vector<vector<string> > development_state_string_sequences;
    if (!development_path_.empty()) {
	ReadLines(development_path_, true,
		  &development_observation_string_sequences,
		  &development_state_string_sequences);
    }
    string temp_model_path = tmpnam(nullptr);
    decoding_method_ = "mbr";  // More appropriate than Viterbi for EM.
    double max_development_accuracy = 0.0;
    size_t no_improvement_count = 0;

    // Start EM iterations.
    Report(util_string::printf_format("\nEM ITERATIONS FOR TRANSITION"));

    // Compute initial accuracy on the development data, save the model.
    if (!development_path_.empty()) {
	vector<vector<string> > predictions;
	for (size_t i = 0;
	     i < development_observation_string_sequences.size(); ++i) {
	    vector<string> prediction;
	    Predict(development_observation_string_sequences[i], &prediction);
	    predictions.push_back(prediction);
	}
	unordered_map<string, string> label_mapping;
	double position_accuracy;
	double sequence_accuracy;
	eval_sequential::compute_accuracy_mapping_labels(
	    development_state_string_sequences, predictions,
	    &position_accuracy, &sequence_accuracy, &label_mapping);

	Report(util_string::printf_format(
		   "Dev data provided: %s   (original accuracy %.2f%%)",
		   util_file::get_file_name(development_path_).c_str(),
		   position_accuracy));
	max_development_accuracy = position_accuracy;
	Save(temp_model_path);
    }

    double log_likelihood = -numeric_limits<double>::infinity();  // Convention.

    // Iterate (100 times maximum) until
    //    1. If dev data is given:  we fail to improve the dev accuracy.
    //    2. If dev data not given: we reach the pre-specified number of EM
    //                              iterations, or we reach a local optimum.
    for (size_t iteration_num = 0; iteration_num < 100; ++iteration_num) {
	// Set up expected counts of state bigrams.
	vector<vector<double> > state_bigram_count(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    state_bigram_count[state].resize(NumStates() + 1, 0.0);  // +stop
	}

	// Pre-compute a matrix of state bigram probabilities.
	Eigen::MatrixXd state_bigram_probabilities =
	    transition_matrix * average_state_probabilities.asDiagonal();
	double new_log_likelihood = 0.0;

	// Go through the observed final counts.
	for (const auto &observation_pair : final_observation_count) {
	    Observation observation = observation_pair.first;
	    double probability_final_observation =
		state_bigram_probabilities.row(StoppingState()).dot(
		    emission_matrix.row(observation));
	    if (probability_final_observation > 0.0) {
		// Only consider "possible" observations.
		size_t final_count = observation_pair.second;
		new_log_likelihood +=
		    final_count * log(probability_final_observation);

		for (State state = 0; state < NumStates(); ++state) {
		    // Accumulate the probability of stopping given the
		    // observation.
		    double state_stopping_probability_given_observation =
			emission_matrix(observation, state) *
			state_bigram_probabilities(StoppingState(), state) /
			probability_final_observation;
		    state_bigram_count[state][StoppingState()] += final_count *
			state_stopping_probability_given_observation;
		}
	    }
	}

	// Go through the observed bigram counts.
	for (const auto &observation_bigram_pair1 : observation_bigram_count) {
	    Observation observation = observation_bigram_pair1.first;

	    // Compute a vector of p(observation, next_state).
	    Eigen::VectorXd next_state_probabilities_with_observation_all =
		state_bigram_probabilities *
		emission_matrix.row(observation).transpose();

	    // Note: This subvector operation assumes that the stopping state
	    //       has the highest index.
	    Eigen::VectorXd next_state_probabilities_with_observation =
		next_state_probabilities_with_observation_all.head(NumStates());

	    for (const auto &observation_bigram_pair2 :
		     observation_bigram_pair1.second) {
		Observation next_observation = observation_bigram_pair2.first;
		double observation_bigram_probability =
		    emission_matrix.row(next_observation).dot(
			next_state_probabilities_with_observation);

		if (observation_bigram_probability > 0.0) {
		    // Only consider "possible" observations.
		    size_t bigram_count = observation_bigram_pair2.second;
		    new_log_likelihood +=
			bigram_count * log(observation_bigram_probability);

		    for (State state = 0; state < NumStates(); ++state) {
			for (State next_state = 0; next_state < NumStates();
			     ++next_state) {
			    // Accumulate the probability of transitioning from
			    // one state to another given the observation
			    // bigram.
			    double
				state_bigram_probability_with_observed_bigram =
				emission_matrix(observation, state) *
				state_bigram_probabilities(next_state, state) *
				emission_matrix(next_observation, next_state);

			    double
				state_bigram_probability_given_observed_bigram =
				state_bigram_probability_with_observed_bigram /
				observation_bigram_probability;
			    state_bigram_count[state][next_state] +=
				bigram_count *
				state_bigram_probability_given_observed_bigram;
			}
		    }
		}
	    }
	}

	// Update the parameters.
	for (State state = 0; state < NumStates(); ++state) {
	    double state_normalizer =
		accumulate(state_bigram_count[state].begin(),
			   state_bigram_count[state].end(), 0.0);
	    if (state_normalizer <= 0.0) {
		// Don't update if this state is not "seen" in the data.
		continue;
	    }
	    for (State next_state = 0; next_state < NumStates() + 1;  // +stop
		 ++next_state) {
		transition_matrix(next_state, state) =
		    state_bigram_count[state][next_state] / state_normalizer;
		transition_[state][next_state] =
		    util_math::log0(transition_matrix(next_state, state));
	    }
	}
	// Must always increase likelihood.
	double likelihood_difference = new_log_likelihood - log_likelihood;
	ASSERT(likelihood_difference > -1e-5, "Likelihood decreased by: "
	       << likelihood_difference);
	log_likelihood = new_log_likelihood;

	// Stopping critera: development accuracy, or likelihood.
	string line = util_string::printf_format(
	    "Iteration %ld: %.2f (%.2f)", iteration_num + 1,
	    new_log_likelihood, likelihood_difference);

	if (!development_path_.empty()) {
	    if ((iteration_num + 1) % development_interval_ != 0) {
		Report(line);
		continue;  // Not at a checking interval.
	    }
	    // Check the accuracy on the (labeled) development data.
	    vector<vector<string> > predictions;
	    for (size_t i = 0;
		 i < development_observation_string_sequences.size(); ++i) {
		vector<string> prediction;
		Predict(development_observation_string_sequences[i],
			&prediction);
		predictions.push_back(prediction);
	    }

	    unordered_map<string, string> label_mapping;
	    double position_accuracy;
	    double sequence_accuracy;
	    eval_sequential::compute_accuracy_mapping_labels(
		development_state_string_sequences, predictions,
		&position_accuracy, &sequence_accuracy, &label_mapping);

	    line += "   " + util_string::printf_format(
		"dev accuracy %.2f%% ", position_accuracy);
	    if (position_accuracy > max_development_accuracy) {  // New record.
		max_development_accuracy = position_accuracy;
		no_improvement_count = 0;  // Reset.
		Save(temp_model_path);
		line += " (new record)";
	    } else {  // No improvement.
		++no_improvement_count;
		line += util_string::printf_format(
		    " (no improvement %d)", no_improvement_count);
		if (no_improvement_count >= max_num_no_improvement_) {
		    Report(line);
		    Load(temp_model_path);
		    break;
		}
	    }
	} else {  // No development data is given.
	    if (iteration_num + 1 >= max_num_em_iterations_transition_ ||
		likelihood_difference < 1e-10) {
		Report(line);
		break;
	    }
	}
	Report(line);
    }
    remove(temp_model_path.c_str());
}

void HMM::ConstructEmissionMatrix(Eigen::MatrixXd *emission_matrix) {
    emission_matrix->resize(NumObservations(), NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	// Each column is a distribution.
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    (*emission_matrix)(observation, state) =
		exp(emission_[state][observation]);
	}
    }
}

void HMM::ConstructTransitionMatrix(Eigen::MatrixXd *transition_matrix) {
    transition_matrix->resize(NumStates() + 1, NumStates());  // +stop
    for (State state = 0; state < NumStates(); ++state) {
	// Each column is a distribution.
	for (State next_state = 0; next_state < NumStates() + 1;  // +stop
	     ++next_state) {
	    (*transition_matrix)(next_state, state) =
		exp(transition_[state][next_state]);
	}
    }
}

void HMM::InitializeTransitionParametersUniformly() {
    transition_.resize(NumStates());
    for (State state1 = 0; state1 < NumStates(); ++state1) {
	transition_[state1].resize(NumStates() + 1);  // +stop
	double state1_normalizer = 0.0;
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    double value = 1.0;  // Any constant.
	    transition_[state1][state2] = value;
	    state1_normalizer += value;
	}
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    transition_[state1][state2] =
		log(transition_[state1][state2]) - log(state1_normalizer);
	}
    }
}

void HMM::RunBaumWelch(const string &data_path) {
    if (emission_.size() == 0 || transition_.size() == 0 ||
	prior_.size() == 0) {  // Initialize parameters randomly.
	InitializeParametersRandomly();
    }

    // Prepare data-driven model development.
    vector<vector<string> > development_observation_string_sequences;
    vector<vector<string> > development_state_string_sequences;
    if (!development_path_.empty()) {
	ReadLines(development_path_, true,
		  &development_observation_string_sequences,
		  &development_state_string_sequences);
    }
    string temp_model_path = tmpnam(nullptr);
    decoding_method_ = "mbr";  // More appropriate than Viterbi for EM.
    double max_development_accuracy = 0.0;
    size_t no_improvement_count = 0;

    // Start EM iterations.
    Report(util_string::printf_format("\nEM ITERATIONS FOR BAUM-WELCH"));

    // Compute initial accuracy on the development data, save the model.
    if (!development_path_.empty()) {
	vector<vector<string> > predictions;
	for (size_t i = 0;
	     i < development_observation_string_sequences.size(); ++i) {
	    vector<string> prediction;
	    Predict(development_observation_string_sequences[i], &prediction);
	    predictions.push_back(prediction);
	}
	unordered_map<string, string> label_mapping;
	double position_accuracy;
	double sequence_accuracy;
	eval_sequential::compute_accuracy_mapping_labels(
	    development_state_string_sequences, predictions,
	    &position_accuracy, &sequence_accuracy, &label_mapping);
	Report(util_string::printf_format(
		   "Dev data provided: %s   (random accuracy %.2f%%)",
		   util_file::get_file_name(development_path_).c_str(),
		   position_accuracy));
	max_development_accuracy = position_accuracy;
	Save(temp_model_path);
    }

    double log_likelihood = -numeric_limits<double>::infinity();  // Convention.

    // Iterate (1000 times maximum) until
    //    1. If dev data is given:  we fail to improve the dev accuracy.
    //    2. If dev data not given: we reach the pre-specified number of
    //                              Baum-Welch iterations, or we reach a local
    //                              optimum.
    for (size_t iteration_num = 0; iteration_num < 1000; ++iteration_num) {
	// Set up expected counts.
	vector<vector<double> > emission_count(NumObservations());
	for (State state = 0; state < NumStates(); ++state) {
	    emission_count[state].resize(NumObservations(), 0.0);
	}
	vector<vector<double> > transition_count(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    transition_count[state].resize(NumStates() + 1, 0.0);  // +stop
	}
	vector<double> prior_count(NumStates(), 0.0);

	// Go through the data to accumulated expected counts.
	double new_log_likelihood = 0.0;
	ifstream data_file(data_path, ios::in);
	ASSERT(data_file.is_open(), "Cannot open " << data_path);
	vector<string> observation_string_sequence;
	vector<string> state_string_sequence;  // Won't be used.
	while (ReadLine(false, &data_file, &observation_string_sequence,
			&state_string_sequence)) {
	    size_t length = observation_string_sequence.size();
	    vector<Observation> observation_sequence;
	    ConvertObservationSequence(observation_string_sequence,
				       &observation_sequence);

	    vector<vector<double> > al;  // Forward probabilities.
	    Forward(observation_sequence, &al);

	    // Calculate the log probability of the observation sequence.
	    double log_probability = -numeric_limits<double>::infinity();
	    for (State state = 0; state < NumStates(); ++state) {
		log_probability = util_math::sum_logs(
		    log_probability, al[length - 1][state] +
		    transition_[state][StoppingState()]);
	    }
	    if (log_probability == -numeric_limits<double>::infinity()) {
		// This line contains an observation sequence that has
		// probability 0 under the parameter values. Just skip it!
		continue;
	    }
	    new_log_likelihood += log_probability;

	    vector<vector<double> > be;  // Backward probabilities.
	    Backward(observation_sequence, &be);

	    // Accumulate initial state probabilities.
	    for (State state = 0; state < NumStates(); ++state) {
		prior_count[state] +=
		    exp(al[0][state] + be[0][state] - log_probability);
	    }

	    for (size_t i = 0; i < length; ++i) {
		Observation observation = observation_sequence[i];

		// Accumulate emission probabilities
		for (State state = 0; state < NumStates(); ++state) {
		    emission_count[state][observation] +=
			exp(al[i][state] + be[i][state] - log_probability);
		    if (i > 0) {
			// Accumulate transition probabilities.
			for (State previous_state = 0;
			     previous_state < NumStates(); ++previous_state) {
			    transition_count[previous_state][state] +=
				exp(al[i - 1][previous_state] +
				    transition_[previous_state][state] +
				    emission_[state][observation] +
				    be[i][state] - log_probability);
			}
		    }
		}
	    }
	    // Accumulate final state probabilities.
	    for (State state = 0; state < NumStates(); ++state) {
		transition_count[state][StoppingState()] +=
		    exp(al[length - 1][state] + be[length - 1][state] -
			log_probability);
	    }
	}

	// Update parameters from the expected counts.
	for (State state = 0; state < NumStates(); ++state) {
	    double state_normalizer = accumulate(emission_count[state].begin(),
						 emission_count[state].end(),
						 0.0);
	    for (Observation observation = 0; observation < NumObservations();
		 ++observation) {
		emission_[state][observation] =
		    log(emission_count[state][observation]) -
		    log(state_normalizer);
	    }
	}
	for (State state1 = 0; state1 < NumStates(); ++state1) {
	    double state1_normalizer =
		accumulate(transition_count[state1].begin(),
			   transition_count[state1].end(), 0.0);
	    for (State state2 = 0; state2 < NumStates() + 1; // +stop
		 ++state2) {
		transition_[state1][state2] =
		    log(transition_count[state1][state2]) -
		    log(state1_normalizer);
	    }
	}
	double prior_normalizer = accumulate(prior_count.begin(),
					     prior_count.end(), 0.0);
	for (State state = 0; state < NumStates(); ++state) {
	    prior_[state] = log(prior_count[state]) - log(prior_normalizer);
	}
	CheckProperDistribution();

	// Must always increase likelihood.
	double likelihood_difference = new_log_likelihood - log_likelihood;
	ASSERT(likelihood_difference > -1e-5, "Likelihood decreased by: "
	       << likelihood_difference);
	log_likelihood = new_log_likelihood;

	// Stopping critera: development accuracy, or likelihood.
	string line = util_string::printf_format(
	    "Iteration %ld: %.2f (%.2f)", iteration_num + 1,
	    new_log_likelihood, likelihood_difference);

	if (!development_path_.empty()) {
	    if ((iteration_num + 1) % development_interval_ != 0) {
		Report(line);
		continue;  // Not at a checking interval.
	    }
	    // Check the accuracy on the (labeled) development data.
	    vector<vector<string> > predictions;
	    for (size_t i = 0;
		 i < development_observation_string_sequences.size(); ++i) {
		vector<string> prediction;
		Predict(development_observation_string_sequences[i],
			&prediction);
		predictions.push_back(prediction);
	    }

	    unordered_map<string, string> label_mapping;
	    double position_accuracy;
	    double sequence_accuracy;
	    eval_sequential::compute_accuracy_mapping_labels(
		development_state_string_sequences, predictions,
		&position_accuracy, &sequence_accuracy, &label_mapping);
	    line += "   " + util_string::printf_format(
		"dev accuracy %.2f%% ", position_accuracy);
	    if (position_accuracy > max_development_accuracy) {
		max_development_accuracy = position_accuracy;
		no_improvement_count = 0;  // Reset.
		Save(temp_model_path);
		line += " (new record)";
	    } else {
		++no_improvement_count;
		line += util_string::printf_format(
		    " (no improvement %d)", no_improvement_count);
		if (no_improvement_count >= max_num_no_improvement_) {
		    Report(line);
		    Load(temp_model_path);
		    break;
		}
	    }
	} else {  // No development data is given.
	    if (iteration_num + 1 >= max_num_em_iterations_baumwelch_ ||
		likelihood_difference < 1e-10) {
		Report(line);
		break;
	    }
	}
	Report(line);
    }
    remove(temp_model_path.c_str());
}

void HMM::InitializeParametersRandomly() {
    ASSERT(NumObservations() > 0 && NumStates() > 0, "Must have dictionaries");
    random_device device;
    default_random_engine engine(device());
    normal_distribution<double> normal(0.0, 1.0);  // Standard Gaussian.

    // Generate emission parameters.
    emission_.resize(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	emission_[state].resize(NumObservations());
	double state_normalizer = 0.0;
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    double value = fabs(normal(engine));
	    emission_[state][observation] = value;
	    state_normalizer += value;
	}
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    emission_[state][observation] =
		log(emission_[state][observation]) - log(state_normalizer);
	}
    }

    // Generate transition parameters.
    transition_.resize(NumStates());
    for (State state1 = 0; state1 < NumStates(); ++state1) {
	transition_[state1].resize(NumStates() + 1);  // +stop
	double state1_normalizer = 0.0;
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    double value = fabs(normal(engine));
	    transition_[state1][state2] = value;
	    state1_normalizer += value;
	}
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    transition_[state1][state2] =
		log(transition_[state1][state2]) - log(state1_normalizer);
	}
    }

    // Generate prior parameters.
    prior_.resize(NumStates());
    double prior_normalizer = 0.0;
    for (State state = 0; state < NumStates(); ++state) {
	double value = fabs(normal(engine));
	prior_[state] = value;
	prior_normalizer += value;
    }
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = log(prior_[state]) - log(prior_normalizer);
    }

    CheckProperDistribution();
}

void HMM::InitializeParametersFromClusters(
    const unordered_map<Observation, unordered_map<Observation, size_t> >
    &observation_bigram_count,
    const unordered_map<Observation, size_t> &initial_observation_count,
    const unordered_map<Observation, size_t> &final_observation_count) {
    ASSERT(NumObservations() > 0 && NumStates() > 0, "Must have dictionaries");
    ASSERT(!cluster_path_.empty(), "Need a cluster file");

    // Read clusters.
    unordered_map<string, unordered_map<string, bool> > cluster_to_observations;
    unordered_map<string, string> observation_to_cluster;
    ifstream cluster_file(cluster_path_, ios::in);
    ASSERT(cluster_file.is_open(), "Cannot open " << cluster_path_);
    while (cluster_file.good()) {
	vector<string> tokens;
	util_file::read_line(&cluster_file, &tokens);
	if (tokens.size() == 0) { continue; }

	// Assume each line has the form: <cluster> <observation> <junk>
	cluster_to_observations[tokens[0]][tokens[1]] = true;
	observation_to_cluster[tokens[1]] = tokens[0];
    }

    // Establish a mapping between clusters and hidden states.
    unordered_map<string, State> cluster_dictionary;
    unordered_map<State, string> cluster_dictionary_inverse;
    for (const auto &cluster_pair : cluster_to_observations) {
	string cluster_string = cluster_pair.first;

	// Note that if there are more clusters than hidden states, some
	// clusters will be mapped to some meaningless numbers.
	cluster_dictionary[cluster_string] = cluster_dictionary.size();
	cluster_dictionary_inverse[cluster_dictionary_inverse.size()] =
	    cluster_string;
    }

    // If given transition counts, calculate maximum-likelihood estimates (MLE)
    // from fixed clusters.
    vector<vector<size_t> > emission_count;
    vector<vector<size_t> > transition_count;
    vector<size_t> prior_count;
    if (!observation_bigram_count.empty() &&
	!initial_observation_count.empty() &&
	!final_observation_count.empty()) {
	emission_count.resize(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    emission_count[state].resize(NumObservations(), 0);
	}
	transition_count.resize(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    transition_count[state].resize(NumStates() + 1, 0);  // +stop
	}
	prior_count.resize(NumStates(), 0);

	// Translate observation bigram counts to cluster co-occurrence counts.
	for (const auto &observation1_pair : observation_bigram_count) {
	    string observation1_string =
		observation_dictionary_inverse_[observation1_pair.first];
	    for (const auto &observation2_pair : observation1_pair.second) {
		string observation2_string =
		    observation_dictionary_inverse_[observation2_pair.first];
		if (observation_to_cluster.find(observation1_string) !=
		    observation_to_cluster.end() &&
		    observation_to_cluster.find(observation2_string) !=
		    observation_to_cluster.end()) {  // Must have clusters.
		    string cluster1_string =
			observation_to_cluster[observation1_string];
		    string cluster2_string =
			observation_to_cluster[observation2_string];
		    State state1 = cluster_dictionary[cluster1_string];
		    State state2 = cluster_dictionary[cluster2_string];
		    emission_count[state1][observation1_pair.first] +=
			observation2_pair.second;  // Aggregate!
		    transition_count[state1][state2] +=
			observation2_pair.second;  // Aggregate!
		}
	    }
	}

	// Likewise for initial states...
	for (const auto &observation_pair : initial_observation_count) {
	    string initial_observation_string =
		observation_dictionary_inverse_[observation_pair.first];
	    if (observation_to_cluster.find(initial_observation_string) !=
		observation_to_cluster.end()) {  // Must have a cluster.
		string initial_cluster_string =
		    observation_to_cluster[initial_observation_string];
		State initial_state =
		    cluster_dictionary[initial_cluster_string];
		prior_count[initial_state] +=
		    observation_pair.second;  // Aggregate!
	    }
	}

	// ... and final states.
	for (const auto &observation_pair : final_observation_count) {
	    string final_observation_string =
		observation_dictionary_inverse_[observation_pair.first];
	    if (observation_to_cluster.find(final_observation_string) !=
		observation_to_cluster.end()) {  // Must have a cluster.
		string final_cluster_string =
		    observation_to_cluster[final_observation_string];
		State final_state = cluster_dictionary[final_cluster_string];
		emission_count[final_state][observation_pair.first] +=
		    observation_pair.second;  // Aggregate!
		transition_count[final_state][StoppingState()] +=
		    observation_pair.second;  // Aggregate!
	    }
	}
    }

    double near_zero = 1e-15;  // Some tiny probability.

    // Assign emission parameters.
    emission_.resize(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	emission_[state].resize(NumObservations());
	double state_normalizer = 0.0;
	if (emission_count.size() > 0) {  // Were given unlabeled data: use MLE.
	    for (Observation observation = 0; observation < NumObservations();
		 ++observation) {
		emission_[state][observation] =
		    emission_count[state][observation];
		state_normalizer += emission_count[state][observation];
	    }
	} else {  // Were not given unlabeled data: use hard-clustering.
	    // See if we have a cluster associated with this state: if so,
	    // identify observation strings "active" in that cluster.
	    unordered_map<string, bool> active;
	    if (cluster_dictionary_inverse.find(state) !=
		cluster_dictionary_inverse.end()) {
		active = cluster_to_observations[
		    cluster_dictionary_inverse[state]];
	    }

	    for (Observation observation = 0; observation < NumObservations();
		 ++observation) {
		string observation_string =
		    observation_dictionary_inverse_[observation];

		// Distribute (most of) the probability mass over active
		// observations uniformly.
		double value = (active.find(observation_string) !=
				active.end()) ?
		    1.0 / active.size() : near_zero;
		emission_[state][observation] = value;
		state_normalizer += value;
	    }
	}
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    emission_[state][observation] =
		log(emission_[state][observation]) - log(state_normalizer);
	}
    }

    // Assign transition parameters.
    transition_.resize(NumStates());
    for (State state1 = 0; state1 < NumStates(); ++state1) {
	transition_[state1].resize(NumStates() + 1);  // +stop
	double state1_normalizer = 0.0;
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    // Use MLE if given unlabeled data, otherwise uniform.
	    double value = (transition_count.size() > 0) ?
		transition_count[state1][state2] : near_zero;
	    transition_[state1][state2] = value;
	    state1_normalizer += value;
	}
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    transition_[state1][state2] = (state1_normalizer > 0) ?
		log(transition_[state1][state2]) - log(state1_normalizer) :
		- log(NumStates() + 1);
	}
    }

    // Assign prior parameters.
    prior_.resize(NumStates());
    double prior_normalizer = 0.0;
    for (State state = 0; state < NumStates(); ++state) {
	// Use MLE if given unlabeled data, otherwise uniform.
	double value = (prior_count.size() > 0) ?
	    prior_count[state] : near_zero;
	prior_[state] = value;
	prior_normalizer += value;
    }
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = (prior_normalizer > 0) ?
	    log(prior_[state]) - log(prior_normalizer) : -log(NumStates());
    }

    CheckProperDistribution();
}

void HMM::CheckProperDistribution() {
    ASSERT(NumObservations() > 0 && NumStates() > 0, "Empty dictionary?");
    for (State state = 0; state < NumStates(); ++state) {
	double state_sum = 0.0;
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    state_sum += exp(emission_[state][observation]);
	}
	ASSERT(fabs(state_sum - 1.0) < 1e-10, "Emission: " << state_sum);
    }

    for (State state1 = 0; state1 < NumStates(); ++state1) {
	double state1_sum = 0.0;
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    state1_sum += exp(transition_[state1][state2]);
	}
	ASSERT(fabs(state1_sum - 1.0) < 1e-10, "Transition: " << state1_sum);
    }

    double prior_sum = 0.0;
    for (State state = 0; state < NumStates(); ++state) {
	prior_sum += exp(prior_[state]);
    }
    ASSERT(fabs(prior_sum - 1.0) < 1e-10, "Prior: " << prior_sum);
}

void HMM::ConstructDictionaries(const string &data_path, bool labeled) {
    unordered_map<Observation, size_t> observation_count;
    ConstructDictionaries(data_path, labeled, &observation_count);
}

void HMM::ConstructDictionaries(const string &data_path, bool labeled,
				unordered_map<Observation, size_t>
				*observation_count) {
    observation_dictionary_.clear();
    observation_dictionary_inverse_.clear();
    state_dictionary_.clear();
    state_dictionary_inverse_.clear();
    observation_count->clear();

    // Get the frequency of observation types. Also get state types if labeled.
    unordered_map<string, size_t> observation_string_count;
    ifstream data_file(data_path, ios::in);
    ASSERT(data_file.is_open(), "Cannot open " << data_path);
    vector<string> observation_string_sequence;
    vector<string> state_string_sequence;
    while (ReadLine(labeled, &data_file, &observation_string_sequence,
		    &state_string_sequence)) {
	for (size_t i = 0; i < observation_string_sequence.size(); ++i) {
	    string observation_string = (lowercase_) ?
		util_string::lowercase(observation_string_sequence[i]) :
		observation_string_sequence[i];
	    ++observation_string_count[observation_string];
	    if (labeled) { AddStateIfUnknown(state_string_sequence[i]); }
	}
    }

    // Only include frequent observation types in the dictionary.
    for (const auto observation_string_pair : observation_string_count) {
	size_t count = observation_string_pair.second;
	string observation_string = (count > rare_cutoff_) ?
	    observation_string_pair.first : corpus::kRareString;
	Observation observation = AddObservationIfUnknown(observation_string);
	(*observation_count)[observation] += count;
    }
}

Observation HMM::AddObservationIfUnknown(const string &observation_string) {
    ASSERT(!observation_string.empty(), "Adding an empty observation string!");
    if (observation_dictionary_.find(observation_string) ==
	observation_dictionary_.end()) {
	Observation observation = observation_dictionary_.size();
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }
    return observation_dictionary_[observation_string];
}

State HMM::AddStateIfUnknown(const string &state_string) {
    ASSERT(!state_string.empty(), "Adding an empty state string!");
    if (state_dictionary_.find(state_string) == state_dictionary_.end()) {
	State state = state_dictionary_.size();
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }
    return state_dictionary_[state_string];
}

void HMM::ConvertObservationSequence(
    const vector<string> &observation_string_sequence,
    vector<Observation> *observation_sequence) {
    ASSERT(observation_dictionary_.size() > 0, "No observation dictionary");
    observation_sequence->clear();
    for (size_t i = 0; i < observation_string_sequence.size(); ++i) {
	string observation_string = (lowercase_) ?
	    util_string::lowercase(observation_string_sequence[i]) :
	    observation_string_sequence[i];
	Observation observation;
	if (observation_dictionary_.find(observation_string) !=
	    observation_dictionary_.end()) {  // In dictionary.
	    observation = observation_dictionary_[observation_string];
	} else if (rare_cutoff_ > 0) {  // Not in dictionary, but have rare.
	    observation = observation_dictionary_[corpus::kRareString];
	} else {  // Not in dictionary, no rare -> unknown.
	    observation = UnknownObservation();
	}
	observation_sequence->push_back(observation);
    }
}

void HMM::ConvertStateSequence(const vector<string> &state_string_sequence,
			       vector<State> *state_sequence) {
    ASSERT(state_dictionary_.size() > 0, "No state dictionary");
    state_sequence->clear();
    for (size_t i = 0; i < state_string_sequence.size(); ++i) {
	string state_string = state_string_sequence[i];
	ASSERT(state_dictionary_.find(state_string) != state_dictionary_.end(),
	       "No state string: " << state_string);
	State state = state_dictionary_[state_string];
	state_sequence->push_back(state);
    }
}

void HMM::ConvertStateSequence(const vector<State> &state_sequence,
			       vector<string> *state_string_sequence) {
    ASSERT(state_dictionary_inverse_.size() > 0, "No state dictionary");
    state_string_sequence->clear();
    for (size_t i = 0; i < state_sequence.size(); ++i) {
	State state = state_sequence[i];
	ASSERT(state_dictionary_inverse_.find(state) !=
	       state_dictionary_inverse_.end(), "No state: " << state);
	string state_string = state_dictionary_inverse_[state];
	state_string_sequence->push_back(state_string);
    }
}

double HMM::Viterbi(const vector<Observation> &observation_sequence,
		    vector<State> *state_sequence) {
    size_t length = observation_sequence.size();

    // chart[i][h] = log( highest probability of the observation sequence and
    //                    any state sequence from position 1 to i, the i-th
    //                    state being h                                        )
    vector<vector<double> > chart(length);
    vector<vector<State> > backpointer(length);
    for (size_t i = 0; i < length; ++i) {
	chart[i].resize(NumStates(), -numeric_limits<double>::infinity());
	backpointer[i].resize(NumStates());
    }

    // Base case.
    Observation initial_observation = observation_sequence[0];
    for (State state = 0; state < NumStates(); ++state) {
	double emission_value = (initial_observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][initial_observation];
	chart[0][state] = prior_[state] + emission_value;
    }

    // Main body.
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	for (State state = 0; state < NumStates(); ++state) {
	    double emission_value = (observation == UnknownObservation()) ?
		-log(NumObservations()) : emission_[state][observation];
	    double max_log_probability = -numeric_limits<double>::infinity();
	    State best_previous_state = 0;
	    for (State previous_state = 0; previous_state < NumStates();
		 ++previous_state) {
		double log_probability = chart[i - 1][previous_state] +
		    transition_[previous_state][state] + emission_value;
		if (log_probability >= max_log_probability) {
		    max_log_probability = log_probability;
		    best_previous_state = previous_state;
		}
	    }
	    chart[i][state] = max_log_probability;
	    backpointer[i][state] = best_previous_state;
	}
    }

    // Maximization over the final state.
    double max_log_probability = -numeric_limits<double>::infinity();
    State best_final_state = 0;
    for (State state = 0; state < NumStates(); ++state) {
	double sequence_log_probability =
	    chart[length - 1][state] + transition_[state][StoppingState()];
	if (sequence_log_probability >= max_log_probability) {
	    max_log_probability = sequence_log_probability;
	    best_final_state = state;
	}
    }
    if (debug_) {
	double answer = ViterbiExhaustive(observation_sequence, state_sequence);
	ASSERT(fabs(answer - max_log_probability) < 1e-8, "Answer: "
	       << answer << ",  Viterbi: " << max_log_probability);
    }

    // Backtrack to recover the best state sequence.
    RecoverFromBackpointer(backpointer, best_final_state, state_sequence);
    return max_log_probability;
}

void HMM::RecoverFromBackpointer(const vector<vector<State> > &backpointer,
				 State best_final_state,
				 vector<State> *state_sequence) {
    state_sequence->resize(backpointer.size());
    (*state_sequence)[backpointer.size() - 1] = best_final_state;
    State current_best_state = best_final_state;
    for (size_t i = backpointer.size() - 1; i > 0; --i) {
	current_best_state = backpointer.at(i)[current_best_state];
	(*state_sequence)[i - 1] = current_best_state;
    }
}

double HMM::ViterbiExhaustive(const vector<Observation> &observation_sequence,
			      vector<State> *state_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible state sequences.
    vector<vector<State> > all_state_sequences;
    vector<State> seed_states;
    PopulateAllStateSequences(seed_states, length, &all_state_sequences);

    // Enumerate each state sequence to find the best one.
    double max_sequence_log_probability = -numeric_limits<double>::infinity();
    size_t best_sequence_index = 0;
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double sequence_log_probability =
	    ComputeLogProbability(observation_sequence, all_state_sequences[i]);
	if (sequence_log_probability >= max_sequence_log_probability) {
	    max_sequence_log_probability = sequence_log_probability;
	    best_sequence_index = i;
	}
    }
    state_sequence->clear();
    for (size_t i = 0; i < length; ++i) {
	state_sequence->push_back(all_state_sequences[best_sequence_index][i]);
    }
    return max_sequence_log_probability;
}

void HMM::PopulateAllStateSequences(const vector<State> &states, size_t length,
				    vector<vector<State> >
				    *all_state_sequences) {
    if (states.size() == length) {
	all_state_sequences->push_back(states);
    } else {
	for (State state = 0; state < NumStates(); ++state) {
	    vector<State> states_appended = states;
	    states_appended.push_back(state);
	    PopulateAllStateSequences(states_appended, length,
				      all_state_sequences);
	}
    }
}

double HMM::ComputeLogProbability(
    const vector<Observation> &observation_sequence,
    const vector<State> &state_sequence) {
    size_t length = observation_sequence.size();
    ASSERT(state_sequence.size() == length, "Lengths not matching");

    Observation initial_observation = observation_sequence[0];
    State initial_state = state_sequence[0];
    double initial_emission_value =
	(initial_observation == UnknownObservation()) ?
	-log(NumObservations()) : emission_[initial_state][initial_observation];

    double sequence_log_probability =
	prior_[initial_state] + initial_emission_value;
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	State state = state_sequence[i];
	double emission_value = (observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][observation];
	sequence_log_probability +=
	    transition_[state_sequence[i - 1]][state] + emission_value;
    }
    sequence_log_probability +=
	transition_[state_sequence[length - 1]][StoppingState()];
    return sequence_log_probability;
}

double HMM::ComputeLogProbability(
    const vector<Observation> &observation_sequence) {
    size_t length = observation_sequence.size();
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    double forward_value = -numeric_limits<double>::infinity();
    for (State state = 0; state < NumStates(); ++state) {
	forward_value = util_math::sum_logs(
	    forward_value, al[length - 1][state] +
	    transition_[state][StoppingState()]);
    }

    if (debug_) {
	double answer = ComputeLogProbabilityExhaustive(observation_sequence);
	vector<vector<double> > be;
	Backward(observation_sequence, &be);
	for (size_t i = 0; i < length; ++i) {
	    double marginal_sum = -numeric_limits<double>::infinity();
	    for (State state = 0; state < NumStates(); ++state) {
		marginal_sum = util_math::sum_logs(
		    marginal_sum, al[i][state] + be[i][state]);
	    }
	    ASSERT(fabs(answer - marginal_sum) < 1e-5, "Answer: "
		   << answer << ",  marginal sum: " << marginal_sum);
	}
    }
    return forward_value;
}

double HMM::ComputeLogProbabilityExhaustive(
    const vector<Observation> &observation_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible state sequences.
    vector<vector<State> > all_state_sequences;
    vector<State> seed_states;
    PopulateAllStateSequences(seed_states, length, &all_state_sequences);

    // Sum over all state sequences.
    double sum_sequence_log_probability = -numeric_limits<double>::infinity();
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double sequence_log_probability =
	    ComputeLogProbability(observation_sequence, all_state_sequences[i]);
	sum_sequence_log_probability = util_math::sum_logs(
	    sum_sequence_log_probability, sequence_log_probability);
    }
    return sum_sequence_log_probability;
}

void HMM::Forward(const vector<Observation> &observation_sequence,
		  vector<vector<double> > *al) {
    size_t length = observation_sequence.size();

    // al[i][h] = log( probability of the observation sequence from position
    //                 1 to i, the i-th state being h                         )
    al->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*al)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base case.
    Observation initial_observation = observation_sequence[0];
    for (State state = 0; state < NumStates(); ++state) {
	double emission_value = (initial_observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][initial_observation];
	(*al)[0][state] = prior_[state] + emission_value;
    }

    // Main body.
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	for (State state = 0; state < NumStates(); ++state) {
	    double emission_value = (observation == UnknownObservation()) ?
		-log(NumObservations()) : emission_[state][observation];
	    double log_summed_probabilities =
		-numeric_limits<double>::infinity();
	    for (State previous_state = 0; previous_state < NumStates();
		 ++previous_state) {
		double log_probability = (*al)[i - 1][previous_state] +
		    transition_[previous_state][state] + emission_value;
		log_summed_probabilities =
		    util_math::sum_logs(log_summed_probabilities,
					log_probability);
	    }
	    (*al)[i][state] = log_summed_probabilities;
	}
    }
}

void HMM::Backward(const vector<Observation> &observation_sequence,
		   vector<vector<double> > *be) {
    size_t length = observation_sequence.size();

    // be[i][h] = log( probability of the observation sequence from position
    //                 i+1 to the end, conditioned on the i-th state being h )
    be->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*be)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base case.
    for (State state = 0; state < NumStates(); ++state) {
	(*be)[length - 1][state] = transition_[state][StoppingState()];
    }

    // Main body.
    for (int i = length - 2; i >= 0; --i) {
	Observation next_observation = observation_sequence[i + 1];
	for (State state = 0; state < NumStates(); ++state) {
	    double log_summed_probabilities =
		-numeric_limits<double>::infinity();
	    for (State next_state = 0; next_state < NumStates(); ++next_state) {
		double emission_value = (next_observation ==
					 UnknownObservation()) ?
		    -log(NumObservations()) :
		    emission_[next_state][next_observation];
		double log_probability = transition_[state][next_state] +
		    emission_value + (*be)[i + 1][next_state];
		log_summed_probabilities =
		    util_math::sum_logs(log_summed_probabilities,
					log_probability);
	    }
	    (*be)[i][state] = log_summed_probabilities;
	}
    }
}

void HMM::ComputeLogMarginal(const vector<Observation> &observation_sequence,
			       vector<vector<double> > *marginal) {
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    vector<vector<double> > be;
    Backward(observation_sequence, &be);

    double log_sequence_probability = -numeric_limits<double>::infinity();
    for (State state = 0; state < NumStates(); ++state) {
	log_sequence_probability = util_math::sum_logs(
	    log_sequence_probability,
	    al[observation_sequence.size() - 1][state] +
	    transition_[state][StoppingState()]);
    }

    marginal->clear();
    marginal->resize(observation_sequence.size());
    for (size_t i = 0; i < observation_sequence.size(); ++i) {
	(*marginal)[i].resize(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    (*marginal)[i][state] = al[i][state] + be[i][state] -
		log_sequence_probability;
	}
    }
}

void HMM::MinimumBayesRisk(const vector<Observation> &observation_sequence,
			   vector<State> *state_sequence) {
    state_sequence->clear();
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    vector<vector<double> > be;
    Backward(observation_sequence, &be);
    for (size_t i = 0; i < observation_sequence.size(); ++i) {
	double max_log_probability = -numeric_limits<double>::infinity();
	State best_state = 0;
	for (State state = 0; state < NumStates(); ++state) {
	    double log_probability = al[i][state] + be[i][state];
	    if (log_probability >= max_log_probability) {
		max_log_probability = log_probability;
		best_state = state;
	    }
	}
	state_sequence->push_back(best_state);
    }
}

void HMM::GreedyDecoding(const vector<Observation> &observation_sequence,
			 vector<State> *state_sequence) {
    state_sequence->clear();

    double max_log_probability = -numeric_limits<double>::infinity();
    State best_state = 0;

    Observation initial_observation = observation_sequence[0];
    for (State state = 0; state < NumStates(); ++state) {
	double emission_value = (initial_observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][initial_observation];
	double log_probability = prior_[state] + emission_value;
	if (log_probability >= max_log_probability) {
	    max_log_probability = log_probability;
	    best_state = state;
	}
    }
    state_sequence->push_back(best_state);

    for (size_t i = 1; i < observation_sequence.size(); ++i) {
	max_log_probability = -numeric_limits<double>::infinity();
	best_state = 0;
	Observation observation = observation_sequence[i];
	State previous_state = state_sequence->at(i-1);
	for (State state = 0; state < NumStates(); ++state) {
	    double emission_value = (observation == UnknownObservation()) ?
		-log(NumObservations()) : emission_[state][observation];
	    double log_probability =
		transition_[previous_state][state] + emission_value;
	    if (log_probability >= max_log_probability) {
		max_log_probability = log_probability;
		best_state = state;
	    }
	}
	state_sequence->push_back(best_state);
    }
}

void HMM::Report(const string &report_string) {
    ofstream log_file(LogPath(), ios::out | ios::app);
    log_file << report_string << endl;
    if (verbose_) { cerr << report_string << endl; }
}
