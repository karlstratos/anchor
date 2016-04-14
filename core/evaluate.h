// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for evaluation.

#ifndef CORE_EVALUATE_H_
#define CORE_EVALUATE_H_

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Functions for evaluating sequences.
namespace eval_sequential {
    // Computes accuracy for sequence predictions.
    void compute_accuracy(const vector<vector<string> > &true_sequences,
			  const vector<vector<string> > &predicted_sequences,
			  double *position_accuracy,
			  double *sequence_accuracy);

    // Computes accuracy for sequence predictions with label mapping.
    void compute_accuracy_mapping_labels(
	const vector<vector<string> > &true_sequences,
	const vector<vector<string> > &predicted_sequences,
	double *position_accuracy, double *sequence_accuracy,
	unordered_map<string, string> *label_mapping);
}  // namespace eval_sequential

// Functions for evaluating lexical represenations. When using vectors, it is
// more efficient to pre-normalize their 2-norms and set: normalized = true.
namespace eval_lexical {
    // Computes correlation on files of scored word pairs.
    void compute_correlation(const vector<string> &scored_word_pairs_paths,
			     const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, bool normalized,
			     vector<size_t> *num_instances,
			     vector<size_t> *num_handled,
			     vector<double> *correlation);

    // Computes correlation on a file of scored word pairs.
    void compute_correlation(const string &scored_word_pairs_path,
			     const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, bool normalized,
			     size_t *num_instances, size_t *num_handled,
			     double *correlation);

    // Computes correlation on scored word pairs: [word1] [word2] [score]
    void compute_correlation(
	const vector<tuple<string, string, double> > &scored_word_pairs,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_handled, double *correlation);

    // Computes accuracy on answering analogy questions in files.
    void compute_analogy_accuracy(
	const vector<string> &analogy_questions_paths,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, vector<size_t> *num_instances,
	vector<size_t> *num_handled, vector<double> *accuracy,
	vector<unordered_map<string, double> > *per_type_accuracy);

    // Computes accuracy on answering analogy questions in a file.
    void compute_analogy_accuracy(
	const string &analogy_questions_path,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_instances, size_t *num_handled,
	double *accuracy, unordered_map<string, double> *per_type_accuracy);

    // Computes accuracy on analogy questions:
    //    [type] [word1] [word2] [word3] [word4]
    void compute_analogy_accuracy(
	const vector<tuple<string, string, string, string, string> >
	&analogy_questions,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_handled, double *accuracy,
	unordered_map<string, double> *per_type_accuracy);

    // Given words w1, w2, v1, returns word v2 (not in {w1, w2, v1}) such that
    // "w1:w2 ~ v1:v2". We must have word vectors for {w1, w2, v1}.
    string infer_analogous_word(string w1, string w2, string v1,
				const unordered_map<string, Eigen::VectorXd>
				&word_vectors, bool normalized);
}  // namespace eval_lexical

#endif  // CORE_EVALUATE_H_
