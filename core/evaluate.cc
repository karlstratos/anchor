// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "evaluate.h"

#include "util.h"

namespace eval_sequential {
    void compute_accuracy(const vector<vector<string> > &true_sequences,
			  const vector<vector<string> > &predicted_sequences,
			  double *position_accuracy,
			  double *sequence_accuracy) {
	size_t num_items = 0;
	size_t num_items_correct = 0;
	size_t num_sequences_correct = 0;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    num_items += true_sequences[i].size();
	    bool entire_sequence_is_correct = true;
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		string true_string = true_sequences[i][j];
		string predicted_string = predicted_sequences[i][j];
		if (predicted_string == true_string) {
		    num_items_correct += 1;
		} else {
		    entire_sequence_is_correct = false;
		}
	    }
	    if (entire_sequence_is_correct) { num_sequences_correct += 1; }
	}
	(*position_accuracy) = ((double) num_items_correct) / num_items * 100;
	(*sequence_accuracy) = ((double) num_sequences_correct) /
	    true_sequences.size() * 100;
    }

    void compute_accuracy_mapping_labels(
	const vector<vector<string> > &true_sequences,
	const vector<vector<string> > &predicted_sequences,
	double *position_accuracy, double *sequence_accuracy,
	unordered_map<string, string> *label_mapping) {
	// Create many-to-one label mapping.
	unordered_map<string, unordered_map<string, size_t> > count_matches;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		++count_matches[predicted_sequences[i][j]][
		    true_sequences[i][j]];
	    }
	}
	for (const auto &predicted_pair: count_matches) {
	    vector<pair<string, size_t> > matches;
	    for (const auto &true_pair: predicted_pair.second) {
		matches.emplace_back(true_pair.first, true_pair.second);
	    }
	    sort(matches.begin(), matches.end(),
		 util_misc::sort_pairs_second<string, size_t,
		 greater<size_t> >());
	    (*label_mapping)[predicted_pair.first] = matches[0].first;
	}

	// Use the mapping to match label sets.
	vector<vector<string> > predicted_sequences_mapped(
	    predicted_sequences.size());
	for (size_t i = 0; i < predicted_sequences.size(); ++i) {
	    predicted_sequences_mapped[i].resize(predicted_sequences[i].size());
	    for (size_t j = 0; j < predicted_sequences[i].size(); ++j) {
		predicted_sequences_mapped[i][j] =
		    (*label_mapping)[predicted_sequences[i][j]];
	    }
	}
	compute_accuracy(true_sequences, predicted_sequences_mapped,
			 position_accuracy, sequence_accuracy);
    }
}  // namespace eval_sequential

namespace eval_lexical {
    void compute_correlation(const vector<string> &scored_word_pairs_paths,
			     const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, bool normalized,
			     vector<size_t> *num_instances,
			     vector<size_t> *num_handled,
			     vector<double> *correlation) {
	num_instances->clear();
	num_handled->clear();
	correlation->clear();
	for (size_t i = 0; i < scored_word_pairs_paths.size(); ++i) {
	    string scored_word_pairs_path = scored_word_pairs_paths[i];
	    size_t its_num_instances = 0;
	    size_t its_num_handled = 0;
	    double its_correlation = 0.0;
	    if (util_file::exists(scored_word_pairs_path)) {
		compute_correlation(scored_word_pairs_path, word_vectors,
				    normalized, &its_num_instances,
				    &its_num_handled, &its_correlation);
	    }
	    num_instances->push_back(its_num_instances);
	    num_handled->push_back(its_num_handled);
	    correlation->push_back(its_correlation);
	}
    }

    void compute_correlation(const string &scored_word_pairs_path,
			     const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, bool normalized,
			     size_t *num_instances, size_t *num_handled,
			     double *correlation) {
	ifstream scored_word_pairs_file(scored_word_pairs_path, ios::in);
	ASSERT(scored_word_pairs_file.is_open(), "Cannot open: "
	       << scored_word_pairs_path);
	vector<tuple<string, string, double> > scored_word_pairs;
	while (scored_word_pairs_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&scored_word_pairs_file, &tokens);
	    if (tokens.size() == 0) { continue; }
	    ASSERT(tokens.size() == 3, "Need [word1] [word2] [score]");
	    string word1 = tokens[0];
	    string word2 = tokens[1];
	    double gold_score = stod(tokens[2]);
	    scored_word_pairs.push_back(make_tuple(word1, word2,
						   gold_score));
	}
	*num_instances = scored_word_pairs.size();
	compute_correlation(scored_word_pairs, word_vectors, normalized,
			    num_handled, correlation);
    }

    void compute_correlation(
	const vector<tuple<string, string, double> > &scored_word_pairs,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_handled, double *correlation) {
	vector<double> gold_scores;
	vector<double> cosine_scores;
	*num_handled = 0;
	for (const auto &word1_word2_score : scored_word_pairs) {
	    string word1 = get<0>(word1_word2_score);
	    string word2 = get<1>(word1_word2_score);
	    double gold_score = get<2>(word1_word2_score);
	    string word1_lowercase = util_string::lowercase(word1);
	    string word2_lowercase = util_string::lowercase(word2);
	    Eigen::VectorXd word1_vector;
	    Eigen::VectorXd word2_vector;

	    // Try to find the original string. If fail, try lowercasing.
	    if (word_vectors.find(word1) != word_vectors.end()) {
		word1_vector = word_vectors.at(word1);
	    } else if (word_vectors.find(word1_lowercase) !=
		       word_vectors.end()) {
		word1_vector = word_vectors.at(word1_lowercase);
	    }
	    if (word_vectors.find(word2) != word_vectors.end()) {
		word2_vector = word_vectors.at(word2);
	    } else if (word_vectors.find(word2_lowercase) !=
		       word_vectors.end()) {
		word2_vector = word_vectors.at(word2_lowercase);
	    }

	    // If we have vectors for both word types, compute similarity.
	    if (word1_vector.size() > 0 && word2_vector.size() > 0) {
		if (!normalized) {
		    word1_vector.normalize();
		    word2_vector.normalize();
		}
		double cosine_score = word1_vector.dot(word2_vector);
		gold_scores.push_back(gold_score);
		cosine_scores.push_back(cosine_score);
		++(*num_handled);
	    }
	}
	*correlation = util_math::compute_spearman(gold_scores, cosine_scores);
    }

    void compute_analogy_accuracy(
	const vector<string> &analogy_questions_paths,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, vector<size_t> *num_instances,
	vector<size_t> *num_handled, vector<double> *accuracy,
	vector<unordered_map<string, double> > *per_type_accuracy) {
	num_instances->clear();
	num_handled->clear();
	accuracy->clear();
	per_type_accuracy->clear();
	for (size_t i = 0; i < analogy_questions_paths.size(); ++i) {
	    string analogy_questions_path = analogy_questions_paths[i];
	    size_t its_num_instances = 0;
	    size_t its_num_handled = 0;
	    double its_accuracy = 0.0;
	    unordered_map<string, double> its_per_type_accuracy;
	    if (util_file::exists(analogy_questions_path)) {
		compute_analogy_accuracy(analogy_questions_path, word_vectors,
					 normalized, &its_num_instances,
					 &its_num_handled, &its_accuracy,
					 &its_per_type_accuracy);
	    }
	    num_instances->push_back(its_num_instances);
	    num_handled->push_back(its_num_handled);
	    accuracy->push_back(its_accuracy);
	    per_type_accuracy->push_back(its_per_type_accuracy);
	}
    }

    void compute_analogy_accuracy(
	const string &analogy_questions_path,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_instances, size_t *num_handled,
	double *accuracy, unordered_map<string, double> *per_type_accuracy) {
	vector<tuple<string, string, string, string, string> >
	analogy_questions;
	ifstream analogy_questions_file(analogy_questions_path, ios::in);
	ASSERT(analogy_questions_file.is_open(), "Cannot open file: "
	       << analogy_questions_path);
	while (analogy_questions_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&analogy_questions_file, &tokens);
	    if (tokens.size() == 0) { continue; }
	    ASSERT(tokens.size() == 4 || tokens.size() == 5,
		   "Need question format: either \"[type] [word1] [word2] "
		   "[word3] [word4]\" or \"[word1] [word2] [word3] [word4]");

	    string question_type = (tokens.size() == 4) ? "ALL" : tokens[0];
	    string w1 = (tokens.size() == 4) ? tokens[0] : tokens[1];
	    string w2 = (tokens.size() == 4) ? tokens[1] : tokens[2];
	    string v1 = (tokens.size() == 4) ? tokens[2] : tokens[3];
	    string v2 = (tokens.size() == 4) ? tokens[3] : tokens[4];
	    analogy_questions.emplace_back(question_type, w1, w2, v1, v2);
	}
	*num_instances = analogy_questions.size();
	compute_analogy_accuracy(analogy_questions, word_vectors, normalized,
				 num_handled, accuracy, per_type_accuracy);
    }

    void compute_analogy_accuracy(
	const vector<tuple<string, string, string, string, string> >
	&analogy_questions,
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	bool normalized, size_t *num_handled, double *accuracy,
	unordered_map<string, double> *per_type_accuracy) {

	// Use only relevant word vectors (for efficiency).
	unordered_map<string, Eigen::VectorXd> word_vectors_subset;
	for (const auto &analogy_question : analogy_questions) {
	    string w1 = get<1>(analogy_question);
	    string w2 = get<2>(analogy_question);
	    string v1 = get<3>(analogy_question);
	    string v2 = get<4>(analogy_question);
	    string w1_lowercase = util_string::lowercase(w1);
	    string w2_lowercase = util_string::lowercase(w2);
	    string v1_lowercase = util_string::lowercase(v1);
	    string v2_lowercase = util_string::lowercase(v2);

	    // Try to find the original string. If fail, try lowercasing.
	    if (word_vectors.find(w1) != word_vectors.end()) {
		word_vectors_subset[w1] = word_vectors.at(w1);
	    } else if (word_vectors.find(w1_lowercase) != word_vectors.end()) {
		word_vectors_subset[w1] = word_vectors.at(w1_lowercase);
	    }
	    if (word_vectors.find(w2) != word_vectors.end()) {
		word_vectors_subset[w2] = word_vectors.at(w2);
	    } else if (word_vectors.find(w2_lowercase) != word_vectors.end()) {
		word_vectors_subset[w2] = word_vectors.at(w2_lowercase);
	    }
	    if (word_vectors.find(v1) != word_vectors.end()) {
		word_vectors_subset[v1] = word_vectors.at(v1);
	    } else if (word_vectors.find(v1_lowercase) != word_vectors.end()) {
		word_vectors_subset[v1] = word_vectors.at(v1_lowercase);
	    }
	    if (word_vectors.find(v2) != word_vectors.end()) {
		word_vectors_subset[v2] = word_vectors.at(v2);
	    } else if (word_vectors.find(v2_lowercase) != word_vectors.end()) {
		word_vectors_subset[v2] = word_vectors.at(v2_lowercase);
	    }
	}
	size_t num_correct = 0;
	*num_handled = 0;
	unordered_map<string, double> per_type_num_handled;
	unordered_map<string, double> per_type_num_correct;
	for (const auto &analogy_question : analogy_questions) {
	    string question_type = get<0>(analogy_question);
	    string w1 = get<1>(analogy_question);
	    string w2 = get<2>(analogy_question);
	    string v1 = get<3>(analogy_question);
	    string v2 = get<4>(analogy_question);
	    if (word_vectors_subset.find(w1) != word_vectors_subset.end() &&
		word_vectors_subset.find(w2) != word_vectors_subset.end() &&
		word_vectors_subset.find(v1) != word_vectors_subset.end() &&
		word_vectors_subset.find(v2) != word_vectors_subset.end()) {
		++(*num_handled);
		++per_type_num_handled[question_type];
		string predicted_v2 = infer_analogous_word(w1, w2, v1,
							   word_vectors_subset,
							   normalized);
		if (predicted_v2 == v2) {
		    ++num_correct;
		    ++per_type_num_correct[question_type];
		}
	    }
	}
	*accuracy = ((double) num_correct) / ((double) *num_handled) * 100.0;
	per_type_accuracy->clear();
	for (const auto &type_pair : per_type_num_handled) {
	    string question_type = type_pair.first;
	    (*per_type_accuracy)[question_type] =
		((double) per_type_num_correct[question_type]) /
		((double) per_type_num_handled[question_type]) * 100.0;
	}
    }

    string infer_analogous_word(string w1, string w2, string v1,
				const unordered_map<string, Eigen::VectorXd>
				&word_vectors, bool normalized) {
	ASSERT(word_vectors.find(w1) != word_vectors.end(), "No " << w1);
	ASSERT(word_vectors.find(w2) != word_vectors.end(), "No " << w2);
	ASSERT(word_vectors.find(v1) != word_vectors.end(), "No " << v1);

	Eigen::VectorXd w1_embedding = word_vectors.at(w1);
	Eigen::VectorXd w2_embedding = word_vectors.at(w2);
	Eigen::VectorXd v1_embedding = word_vectors.at(v1);
	if (!normalized) {
	    w1_embedding.normalize();
	    w2_embedding.normalize();
	    v1_embedding.normalize();
	}

	// Use the method of Levy and Goldberg (2014) to compute:
	//                                 scos(w2,v) * scos(v1,v)
	//    v2 = argmax_{v != w1,w2,v1}  ----------------------
	//                                   scos(w1,v) + 0.01
	string predicted_v2 = "";
	double max_score = -numeric_limits<double>::max();
	for (const auto &word_vector_pair : word_vectors) {
	    string word = word_vector_pair.first;
	    if (word == w1 || word == w2 || word == v1) { continue; }
	    Eigen::VectorXd word_embedding = word_vector_pair.second;
	    if (!normalized) { word_embedding.normalize(); }
	    double shifted_cos_w1 =
		(word_embedding.dot(w1_embedding) + 1.0) / 2.0;
	    double shifted_cos_w2 =
		(word_embedding.dot(w2_embedding) + 1.0) / 2.0;
	    double shifted_cos_v1 =
		(word_embedding.dot(v1_embedding) + 1.0) / 2.0;
	    double score =
		shifted_cos_w2 * shifted_cos_v1 / (shifted_cos_w1 + 0.001);
	    if (score > max_score) {
		max_score = score;
		predicted_v2 = word;
	    }
	}
	ASSERT(!predicted_v2.empty(), "No answer for \"" << w1 << ":" << w2
	       << " as in " << v1 << ":" << "?\"");
	return predicted_v2;
    }
}  // namespace eval_lexical
