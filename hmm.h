// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// An implementation of hidden Markove models (HMMs).

#ifndef HIDDEN_MARKOV_MODEL_HMM_H_
#define HIDDEN_MARKOV_MODEL_HMM_H_

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/corpus.h"

using namespace std;

typedef size_t Observation;
typedef size_t State;

class HMM {
public:
    // Initializes empty.
    HMM() { }

    // Initializes with an output directory.
    HMM(const string &output_directory) {
	SetOutputDirectory(output_directory);
    }

    // Initializes randomly.
    HMM(size_t num_observations, size_t num_states) {
	CreateRandomly(num_observations, num_states);
    }

    // Sets the output directory.
    void SetOutputDirectory(const string &output_directory);

    // Resets the content in the output directory.
    void ResetOutputDirectory();

    // Clears the model.
    void Clear();

    // Creates a random HMM.
    void CreateRandomly(size_t num_observations, size_t num_states);

    // Saves HMM parameters to the default model file.
    void Save() { Save(ModelPath()); }

    // Saves HMM parameters to a model file.
    void Save(const string &model_path);

    // Loads HMM parameters from the default model file.
    void Load() { Load(ModelPath()); }

    // Loads HMM parameters from a model file.
    void Load(const string &model_path);

    // Writes useful model information to the default info file.
    void WriteModelInfo() { WriteModelInfo(ModelInfoPath()); }

    // Writes useful model information to a file.
    void WriteModelInfo(const string &info_path);

    // Trains HMM parameters from a text file of labeled sequences.
    void TrainSupervised(const string &data_path);

    // Trains HMM parameters from a text file of unlabeled sequences.
    void TrainUnsupervised(const string &data_path, size_t num_states);

    // Evaluates the sequence labeling accuracy of the HMM on a labeled dataset,
    // writes predictions in a file (if prediction_path != "").
    void Evaluate(const string &labeled_data_path,
		  const string &prediction_path);

    // Predicts a state sequence.
    void Predict(const vector<string> &observation_string_sequence,
		 vector<string> *state_string_sequence);

    // Computes the log probability of the observation string sequence.
    double ComputeLogProbability(
	const vector<string> &observation_string_sequence);

    // Computes the log probability of the observation string sequences.
    double ComputeLogProbability(
	const vector<vector<string> > &observation_string_sequences);

    // Computes the log marginal probabilities for each position:
    //    marginal[i][h] = log(probability of state h at the i-th position,
    //                         given the observation string sequence)
    void ComputeLogMarginal(const vector<string> &observation_string_sequence,
			    vector<vector<double> > *marginal);

    // Reads lines from a data file.
    void ReadLines(const string &file_path, bool labeled,
		   vector<vector<string> > *observation_string_sequences,
		   vector<vector<string> > *state_string_sequences);

    // Reads a line from a data file. Returns true if success, false if there is
    // no more non-empty line: while (ReadLine(...)) { /* process line */ }
    bool ReadLine(bool labeled, ifstream *file,
		  vector<string> *observation_string_sequence,
		  vector<string> *state_string_sequence);

    // Returns the emission probability.
    double EmissionProbability(string state_string, string observation_string);

    // Returns the transition probability.
    double TransitionProbability(string state1_string, string state2_string);

    // Returns the prior probability.
    double PriorProbability(string state_string);

    // Returns the stopping probability.
    double StoppingProbability(string state_string);

    // Returns the number of observation types.
    size_t NumObservations() { return observation_dictionary_.size(); }

    // Returns the number of state types.
    size_t NumStates() { return state_dictionary_.size(); }

    // Returns the index corresponding to an observation string.
    Observation GetObservationIndex(string observation_string);

    // Returns the string corresponding to an observation index.
    string GetObservationString(Observation observation);

    // Returns the index corresponding to a state string.
    State GetStateIndex(string state_string);

    // Returns the string corresponding to a state index.
    string GetStateString(State state);

    // Sets the output directory.
    void set_output_directory(string output_directory) {
	output_directory_ = output_directory;
    }

    // Sets the flag for lowercasing all observation strings.
    void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

    // Sets the rare cutoff.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets the unsupervised learning method.
    void set_unsupervised_learning_method(string unsupervised_learning_method) {
	unsupervised_learning_method_ = unsupervised_learning_method;
    }

    // Sets the maximum number of EM iterations in the Baum-Welch algorithm.
    void set_max_num_em_iterations_baumwelch(
	size_t max_num_em_iterations_baumwelch) {
	max_num_em_iterations_baumwelch_ = max_num_em_iterations_baumwelch;
    }

    // Sets the maximum number of EM iterations for estimating the transition
    // parameters given emission parameters.
    void set_max_num_em_iterations_transition(
	size_t max_num_em_iterations_transition) {
	max_num_em_iterations_transition_ = max_num_em_iterations_transition;
    }

    // Sets the maximum number of Frank-Wolfe iterations.
    void set_max_num_fw_iterations(size_t max_num_fw_iterations) {
	max_num_fw_iterations_ = max_num_fw_iterations;
    }

    // Sets the interval to check development accuracy.
    void set_development_interval(size_t development_interval) {
	development_interval_ = development_interval;
    }

    // Sets the maximum number of iterations without improvement before
    // stopping.
    void set_max_num_no_improvement(size_t max_num_no_improvement) {
	max_num_no_improvement_ = max_num_no_improvement;
    }

    // Sets the context window size.
    void set_window_size(size_t window_size) { window_size_ = window_size; }

    // Sets the context definition.
    void set_context_definition(string context_definition) {
	context_definition_ = context_definition;
    }

    // Sets the convex hull method.
    void set_convex_hull_method(string convex_hull_method) {
	convex_hull_method_ = convex_hull_method;
    }

    // Sets the context extension.
    void set_context_extension(string context_extension) {
	context_extension_ = context_extension;
    }

    // Sets the additive smoothing value.
    void set_add_smooth(double add_smooth) { add_smooth_ = add_smooth; }

    // Sets the power smoothing value.
    void set_power_smooth(double power_smooth) { power_smooth_ = power_smooth; }

    // Sets the number of anchor candidates.
    void set_num_anchor_candidates(size_t num_anchor_candidates) {
	num_anchor_candidates_ = num_anchor_candidates;
    }

    // Sets the weight for new context features.
    void set_extension_weight(double extension_weight) {
	extension_weight_ = extension_weight;
    }

    // Sets the path to development data.
    void set_development_path(string development_path) {
	development_path_ = development_path;
    }

    // Sets the path to clusters.
    void set_cluster_path(string cluster_path) {
	cluster_path_ = cluster_path;
    }

    // Sets the path to anchors.
    void set_anchor_path(string anchor_path) {
	anchor_path_ = anchor_path;
    }

    // Sets the flag for doing post-training local search.
    void set_post_training_local_search(bool post_training_local_search) {
	post_training_local_search_ = post_training_local_search;
    }

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Sets whether to turn on the debug mode.
    void set_debug(bool debug) { debug_ = debug; }

private:
    // Returns the index corresponding to an unknown observation.
    Observation UnknownObservation() { return NumObservations(); }

    // Returns the index corresponding to a special stopping state.
    State StoppingState() { return NumStates(); }

    // Builds a convex hull of observation vectors as rows of a matrix.
    void BuildConvexHull(Corpus *corpus, Eigen::MatrixXd *convex_hull);

    // Extends the context space (i.e., add additional columns to the
    // observation-context co-occurrence matrix).
    void ExtendContextSpace(unordered_map<string, Context> *context_dictionary,
			    unordered_map<Context, unordered_map<Observation,
			    double> > *context_observation_count);

    // Computes the "flipped" emission distributions p(State|Observation) as
    // rows of a matrix by decomposing the given convex hull.
    void ComputeFlippedEmission(const Eigen::MatrixXd &convex_hull,
				const unordered_map<Observation, size_t>
				&observation_count,
				Eigen::MatrixXd *flipped_emission);

    // Find anchor observations.
    void FindAnchors(const Eigen::MatrixXd &convex_hull,
		     const unordered_map<Observation, size_t>
		     &observation_count,
		     vector<Observation> *anchor_observations);

    // Recovers the model parameters given the "flipped" emission parameters.
    void RecoverParametersGivenFlippedEmission(
	const Eigen::MatrixXd &flipped_emission,
	const unordered_map<Observation, size_t> &observation_count,
	const unordered_map<Observation, unordered_map<Observation, size_t> >
	&observation_bigram_count,
	const unordered_map<Observation, size_t> &initial_observation_count,
	const unordered_map<Observation, size_t> &final_observation_count);

    // Recovers the emission parameters given flipped emission.
    void RecoverEmissionParametersGivenFlippedEmission(
	const unordered_map<Observation, size_t> &observation_count,
	const Eigen::MatrixXd &flipped_emission);

    // Recovers the prior parameters given flipped emission.
    void RecoverPriorParametersGivenEmission(
	const unordered_map<Observation, size_t> &initial_observation_count);

    // Recovers the transition parameters given the rest of other parameters.
    void RecoverTransitionParametersGivenOthers(
	const Eigen::VectorXd &average_state_probabilities,
	const unordered_map<Observation, unordered_map<Observation, size_t> >
	&observation_bigram_count,
	const unordered_map<Observation, size_t> &final_observation_count);

    // Organizes the emission parameters into a probability matrix.
    void ConstructEmissionMatrix(Eigen::MatrixXd *emission_matrix);

    // Organizes the transition parameters into a probability matrix.
    void ConstructTransitionMatrix(Eigen::MatrixXd *transition_matrix);

    // Initializes the transition parameters uniformly.
    void InitializeTransitionParametersUniformly();

    // Runs the Baum-Welch algorithm (must already have dictionaries).
    // - If parameters exist, simply start from them.
    // - Otherwise, initialize them randomly.
    void RunBaumWelch(const string &data_path);

    // Initializes parameters randomly (must already have dictionaries).
    void InitializeParametersRandomly();

    // Initializes parameters from clusters (must already have dictionaries).
    // Use MLE if transition counts are non-empty. Otherwise, use uniform
    // distribution except for emission parameters as follows:
    //    A state that has a cluster assigns (nearly all) its probability mass
    //    uniformly to its observation members.
    void InitializeParametersFromClusters(
	const unordered_map<Observation, unordered_map<Observation, size_t> >
	&observation_bigram_count,
	const unordered_map<Observation, size_t> &initial_observation_count,
	const unordered_map<Observation, size_t> &final_observation_count);

    void InitializeParametersFromClusters(const string &cluster_path) {
	InitializeParametersFromClusters(
	    unordered_map<Observation, unordered_map<Observation, size_t> >(),
	    unordered_map<Observation, size_t>(),
	    unordered_map<Observation, size_t>());
    }

    // Check if parameters form proper distributions.
    void CheckProperDistribution();

    // Constructs observation (and state, if labeled) dictionaries.
    void ConstructDictionaries(const string &data_path, bool labeled);

    // Constructs observation (and state, if labeled) dictionaries. Also
    // counts filtered observation types.
    void ConstructDictionaries(const string &data_path, bool labeled,
			       unordered_map<Observation, size_t>
			       *observation_count);

    // Adds the observation string to the dictionary if not already known.
    Observation AddObservationIfUnknown(const string &observation_string);

    // Adds the state string to the dictionary if not already known.
    State AddStateIfUnknown(const string &state_string);

   // Converts an observation sequence from strings to indices.
    void ConvertObservationSequence(
	const vector<string> &observation_string_sequence,
	vector<Observation> *observation_sequence);

    // Converts a state sequence from strings to indices.
    void ConvertStateSequence(const vector<string> &state_string_sequence,
			      vector<State> *state_sequence);

    // Converts a state sequence from indices to strings.
    void ConvertStateSequence(const vector<State> &state_sequence,
			      vector<string> *state_string_sequence);

   // Performs Viterbi decoding, returns the computed probability.
    double Viterbi(const vector<Observation> &observation_sequence,
		   vector<State> *state_sequence);

    // Recovers the best state sequence from the backpointer.
    void RecoverFromBackpointer(const vector<vector<State> > &backpointer,
				State best_final_state,
				vector<State> *state_sequence);

   // Performs exhaustive Viterbi decoding, returns the computed probability.
    double ViterbiExhaustive(const vector<Observation> &observation_sequence,
			     vector<State> *state_sequence);

    // Populates a vector of all state sequences.
    void PopulateAllStateSequences(const vector<State> &states, size_t length,
				   vector<vector<State> > *all_state_sequences);

    // Computes the log probability of the observation/state sequence pair.
    double ComputeLogProbability(
	const vector<Observation> &observation_sequence,
	const vector<State> &state_sequence);

    // Computes the log probability of the observation sequence.
    double ComputeLogProbability(
	const vector<Observation> &observation_sequence);

    // Computes the log probability of the observation sequence exhaustively.
    double ComputeLogProbabilityExhaustive(
	const vector<Observation> &observation_sequence);

    // Computes forward probabilities:
    //    al[i][h] = log(probability of the observation sequence from position
    //                   1 to i, the i-th state being h)
    void Forward(const vector<Observation> &observation_sequence,
		 vector<vector<double> > *al);

    // Computes backward probabilities:
    //    be[i][h] = log(probability of the observation sequence from position
    //                   i+1 to the end, conditioned on the i-th state being h)
    void Backward(const vector<Observation> &observation_sequence,
		  vector<vector<double> > *be);

    // Computes the log marginal probabilities for each position:
    //    marginal[i][h] = log(probability of state h at the i-th position,
    //                         given the observation sequence)
    void ComputeLogMarginal(const vector<Observation> &observation_sequence,
			    vector<vector<double> > *marginal);

    // Performs minimum Bayes risk (MBR) decoding.
    void MinimumBayesRisk(const vector<Observation> &observation_sequence,
			  vector<State> *state_sequence);

    // Performs greedy decoding.
    void GreedyDecoding(const vector<Observation> &observation_sequence,
			vector<State> *state_sequence);

    // Reports status in a log file and optionally the standard output.
    void Report(const string &report_string);

    // Returns the path to the model file.
    string ModelPath() { return output_directory_ + "/model.bin"; }

    // Returns the path to the log file.
    string LogPath() { return output_directory_ + "/log.txt"; }

    // Returns the path to the model info file.
    string ModelInfoPath() { return output_directory_ + "/info.txt"; }

    // Returns the path to the flipped emission parameters file.
    string FlippedEmissionPath() {
	return output_directory_ + "/flipped_emission.txt";
    }

    // Special string for separating observation/state in data files.
    const string kObservationStateSeperator_ = "__<label>__";

    // Special string for representing synthetic states.
    const string kState_ = "state";

    // Maps an observation string to a unique index.
    unordered_map<string, Observation> observation_dictionary_;

    // Maps an observation index to its original string form.
    unordered_map<Observation, string> observation_dictionary_inverse_;

    // Maps a state string to a unique index.
    unordered_map<string, State> state_dictionary_;

    // Maps a state index to its original string form.
    unordered_map<State, string> state_dictionary_inverse_;

    // Emission log probabilities.
    vector<vector<double> > emission_;

    // Transition log probabilities.
    vector<vector<double> > transition_;

    // Prior log probabilities.
    vector<double> prior_;

    // Path to the output directory.
    string output_directory_;

    // Lowercase all observation strings?
    bool lowercase_ = false;

    // Observation types that occur <= this number in the training data are
    // considered as a single symbol (corpus::kRareString).
    size_t rare_cutoff_ = 0;

    // Unsupervised learning method.
    string unsupervised_learning_method_ = "bw";

    // Maximum number of EM iterations in the Baum-Welch algorithm.
    size_t max_num_em_iterations_baumwelch_ = 1000;

    // Maximum number of EM iterations for estimating the transition parameters
    // given emission parameters.
    size_t max_num_em_iterations_transition_ = 1;

    // Maximum number of Frank-Wolfe iterations.
    size_t max_num_fw_iterations_ = 1000;

    // Interval to check development accuracy.
    size_t development_interval_ = 10;

    // Maximum number of iterations without improvement before stopping.
    size_t max_num_no_improvement_ = 10;

    // Size of the sliding window (odd => symmetric, even => assymmetric).
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "list";

    // Convex hull method.
    string convex_hull_method_ = "brown";

    // Context extension.
    string context_extension_ = "";

    // Additive smoothing value.
    double add_smooth_ = 10.0;

    // Power smoothing value.
    double power_smooth_ = 0.5;

    // Number of anchor candidates (most frequent observation types).
    size_t num_anchor_candidates_ = 300;

    // Weight for new context features v: weight * ||u|| = ||v|| (l2 norm).
    double extension_weight_ = 0.01;

    // Path to development data.
    string development_path_;

    // Path to clusters.
    string cluster_path_;

    // Path to anchors.
    string anchor_path_;

    // Do post-training local search?
    bool post_training_local_search_ = false;

    // Decoding method.
    string decoding_method_ = "mbr";

    // Print messages to stderr?
    bool verbose_ = true;

    // Turn on the debug mode?
    bool debug_ = false;
};

#endif  // HIDDEN_MARKOV_MODEL_HMM_H_
