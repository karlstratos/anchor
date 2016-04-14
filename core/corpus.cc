// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "corpus.h"

#include <iomanip>
#include <limits>
#include <random>

namespace corpus {
    void decompose(SMat matrix, size_t desired_rank,
		   const string &transformation_method, double add_smooth,
		   double power_smooth,  double context_power_smooth,
		   const string &scaling_method,
		   Eigen::MatrixXd *left_singular_vectors,
		   Eigen::MatrixXd *right_singular_vectors,
		   Eigen::VectorXd *singular_values) {
	// Get the number of word/context samples by summing columns/rows.
	unordered_map<Word, double> num_word_samples;  // #(w)
	unordered_map<Context, double> num_context_samples;  // #(c)
	sparsesvd::sum_rows_columns(matrix, &num_word_samples,
				    &num_context_samples);
	size_t num_word_types = num_word_samples.size();
	size_t num_context_types = num_context_samples.size();

	// 1. Transform each aggregate count and compute the new sum.
	double word_context_normalizer = 0.0;  // sum_(w,c) {transformed #(w,c)}
	for (Context c = 0; c < num_context_types; ++c) {
	    size_t current_nonzero_index = matrix->pointr[c];
	    size_t next_start_nonzero_index = matrix->pointr[c + 1];
	    while (current_nonzero_index < next_start_nonzero_index) {
		// [*] Do not apply additive smoothing to co-occurrence counts.
		matrix->value[current_nonzero_index] =
		    transform(matrix->value[current_nonzero_index], 0,  // [*]
			      power_smooth, transformation_method);
		word_context_normalizer += matrix->value[current_nonzero_index];
		++current_nonzero_index;
	    }
	}
	double word_normalizer = 0.0;  // sum_w {transformed #(w)}
	for (Word w = 0; w < num_word_types; ++w) {
	    num_word_samples[w] = transform(num_word_samples[w], add_smooth,
					    power_smooth,
					    transformation_method);
	    word_normalizer += num_word_samples[w];

	}
	double context_normalizer = 0.0;  // sum_c {transformed #(c)}
	for (Context c = 0; c < num_context_types; ++c) {
	    // [*] Optionally do power smoothing for context distributions.
	    num_context_samples[c] =
		transform(num_context_samples[c], add_smooth,
			  power_smooth * context_power_smooth,  // [*]
			  transformation_method);
	    context_normalizer += num_context_samples[c];
	}

	// 2. Scale each transformed #(w,c) by #(w), #(c), or not.
	for (Context c = 0; c < num_context_types; ++c) {
	    size_t current_nonzero_index = matrix->pointr[c];
	    size_t next_start_nonzero_index = matrix->pointr[c + 1];
	    while (current_nonzero_index < next_start_nonzero_index) {
		Word w = matrix->rowind[current_nonzero_index];
		if (scaling_method == "none") {  // No scaling.
		} else if (scaling_method == "ppmi") {
		    // Positive pointwise mutual information scaling:
		    //    max(log p(w,c) - log p(w) - log p(c), 0)
		    double pmi = log(matrix->value[current_nonzero_index]);
		    pmi -= log(num_word_samples[w]);
		    pmi -= log(num_context_samples[c]);
		    pmi += log(word_normalizer);
		    pmi += log(context_normalizer);
		    pmi -= log(word_context_normalizer);
		    matrix->value[current_nonzero_index] = max(pmi, 0.0);
		} else if (scaling_method == "reg") {
		    // Regression scaling.
		    matrix->value[current_nonzero_index] /= num_word_samples[w];
		} else if (scaling_method == "cca") {
		    // Canonical correlation analysis scaling:
		    //    p(w,c) / sqrt{p(w)} / sqrt{p(c)}
		    double cca_value = matrix->value[current_nonzero_index];
		    cca_value /= sqrt(num_word_samples[w]);
		    cca_value /= sqrt(num_context_samples[c]);
		    cca_value *= sqrt(word_normalizer);
		    cca_value *= sqrt(context_normalizer);
		    cca_value /= word_context_normalizer;
		    matrix->value[current_nonzero_index] = cca_value;
		}
		++current_nonzero_index;
	    }
	}

	// 3. Perform a low-rank SVD.
	size_t actual_rank;
	sparsesvd::compute_svd(matrix, desired_rank,
			       left_singular_vectors, right_singular_vectors,
			       singular_values, &actual_rank);
    }

    void decompose(SMat matrix, size_t desired_rank,
		   const string &transformation_method, double add_smooth,
		   double power_smooth, const string &scaling_method,
		   Eigen::MatrixXd *left_singular_vectors,
		   Eigen::MatrixXd *right_singular_vectors,
		   Eigen::VectorXd *singular_values) {
	decompose(matrix, desired_rank, transformation_method, add_smooth,
		  power_smooth, 1.0, scaling_method, left_singular_vectors,
		  right_singular_vectors, singular_values);
    }


    double transform(double count_value, double add_smooth,
		     double power_smooth, string transformation_method) {
	double transformed_value = count_value + add_smooth;
	if (transformation_method == "power") {
	    // Power transform (no transform if power is 1).
	    transformed_value = pow(transformed_value, power_smooth);
	} else if (transformation_method == "log") {
	    // Log transform (add 1 to account for zero-valued entries).
	    transformed_value = log(1 + transformed_value);
	} else {
	    ASSERT(false, "Unknown transformation: " << transformation_method);
	}
	return transformed_value;
    }

    void load_sorted_word_types(size_t rare_cutoff,
				const string &sorted_word_types_path,
				vector<pair<string, size_t> >
				*sorted_word_types) {
	sorted_word_types->clear();
	ifstream sorted_word_types_file(sorted_word_types_path, ios::in);
	ASSERT(sorted_word_types_file.is_open(), "Cannot open file: "
	       << sorted_word_types_path);
	size_t rare_count = 0;
	while (sorted_word_types_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&sorted_word_types_file, &tokens);
	    if (tokens.size() == 0 ) { continue; }
	    string word_type = tokens[0];
	    size_t word_count = stol(tokens[1]);
	    if (word_count > rare_cutoff) {
		(*sorted_word_types).emplace_back(word_type, word_count);
	    } else {
		rare_count += word_count;
	    }
	}
	if (rare_count > 0) {
	    (*sorted_word_types).emplace_back(kRareString, rare_count);
	}
	sort(sorted_word_types->begin(), sorted_word_types->end(),
	     util_misc::sort_pairs_second<string, size_t, greater<size_t> >());
    }

    void load_word_vectors(const string &word_vectors_path,
			   unordered_map<string, Eigen::VectorXd>
			   *word_vectors, bool normalize) {
	word_vectors->clear();
	ifstream word_vectors_file(word_vectors_path, ios::in);
	while (word_vectors_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&word_vectors_file, &tokens);
	    if (tokens.size() == 0) { continue; }

	    // line = [count] [word_string] [value_{1}] ... [value_{dim_}]
	    Eigen::VectorXd vector(tokens.size() - 2);
	    for (size_t i = 0; i < tokens.size() - 2; ++i) {
		vector(i) = stod(tokens[i + 2]);
	    }
	    if (normalize) { vector.normalize(); }
	    (*word_vectors)[tokens[1]] = vector;
	}
    }

    void load_sorted_word_vectors(const string &sorted_word_vectors_path,
				  vector<size_t> *sorted_word_counts,
				  vector<string> *sorted_word_strings,
				  vector<Eigen::VectorXd> *sorted_word_vectors,
				  bool normalize) {
	sorted_word_counts->clear();
	sorted_word_strings->clear();
	sorted_word_vectors->clear();
	ifstream sorted_word_vectors_file(sorted_word_vectors_path, ios::in);
	while (sorted_word_vectors_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&sorted_word_vectors_file, &tokens);
	    if (tokens.size() == 0) { continue; }

	    // line = [count] [word_string] [value_{1}] ... [value_{dim_}]
	    Eigen::VectorXd vector(tokens.size() - 2);
	    for (size_t i = 0; i < tokens.size() - 2; ++i) {
		vector(i) = stod(tokens[i + 2]);
	    }
	    if (normalize) { vector.normalize(); }
	    sorted_word_counts->push_back(stol(tokens[0]));
	    sorted_word_strings->push_back(tokens[1]);
	    sorted_word_vectors->push_back(vector);
	}
    }
}  // namespace corpus

void Corpus::WriteWords(size_t rare_cutoff,
			const string &sorted_word_types_path,
			const string &word_dictionary_path, size_t *num_words,
			size_t *num_word_types, size_t *vocabulary_size) {
    // 1. Write sorted word types.
    unordered_map<string, size_t> word_count;
    (*num_words) = CountWords(&word_count);
    (*num_word_types) = word_count.size();
    vector<pair<string, size_t> > sorted_word_types(word_count.begin(),
						    word_count.end());
    sort(sorted_word_types.begin(), sorted_word_types.end(),
	 util_misc::sort_pairs_second<string, size_t, greater<size_t> >());

    ofstream sorted_word_types_file(sorted_word_types_path, ios::out);
    for (size_t i = 0; i < sorted_word_types.size(); ++i) {
	sorted_word_types_file << sorted_word_types[i].first << " "
			       << sorted_word_types[i].second << endl;
    }

    // 2. Write a word dictionary.
    unordered_map<string, size_t> word_dictionary;
    BuildWordDictionary(word_count, rare_cutoff, &word_dictionary);
    (*vocabulary_size) = word_dictionary.size();
    util_file::binary_write(word_dictionary, word_dictionary_path);
}

void Corpus::WriteContexts(const unordered_map<string, Word> &word_dictionary,
			   bool sentence_per_line,
			   const string &context_definition, size_t window_size,
			   size_t hash_size,
			   const string &context_dictionary_path,
			   const string &context_word_count_path,
			   size_t *num_nonzeros) {
    unordered_map<string, Context> context_dictionary;
    unordered_map<Context, unordered_map<Word, double> > context_word_count;
    *num_nonzeros = SlideWindow(word_dictionary, sentence_per_line,
				context_definition, window_size, hash_size,
				&context_dictionary, &context_word_count);
    util_file::binary_write(context_dictionary, context_dictionary_path);
    sparsesvd::binary_write_sparse_matrix(context_word_count,
					  context_word_count_path);
}

void Corpus::WriteTransitions(
    const unordered_map<string, Word> &word_dictionary,
    const string &bigram_count_path,
    const string &start_count_path,
    const string &end_count_path) {
    unordered_map<Word, unordered_map<Word, size_t> > bigram_count;
    unordered_map<Word, size_t> start_count;
    unordered_map<Word, size_t> end_count;
    CountTransitions(word_dictionary, &bigram_count, &start_count, &end_count);

    util_file::binary_write_primitive(bigram_count, bigram_count_path);
    util_file::binary_write_primitive(start_count, start_count_path);
    util_file::binary_write_primitive(end_count, end_count_path);
}

size_t Corpus::CountWords(unordered_map<string, size_t> *word_count) {
    word_count->clear();
    vector<string> file_list;
    util_file::list_files(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = util_file::get_num_lines(file_path);
	if (verbose_) {
	    cerr << "Counting words in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    ++line_num;
	    util_file::read_line(&file, &word_strings);
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    for (string word_string : word_strings) {
		if (Skip(word_string)) { continue; }
		if (lowercase_) {
		    word_string = util_string::lowercase(word_string);
		}
		++(*word_count)[word_string];

		// If the vocabulary is too large, subtract by the median count
		// and eliminate at least half of the word types.
		if (word_count->size() >= max_vocabulary_size_) {
		    if (verbose_) {
			cerr << endl << "VOCAB SIZE " << word_count->size()
			     << ", SUBTRACTING BY THE MEDIAN COUNT!" << endl;
		    }
		    util_misc::subtract_by_median(word_count);
		}
	    }
	    if (line_num / num_lines >= portion_so_far) {
		portion_so_far += kReportInterval_;
		if (verbose_) { cerr << "." << flush; }
	    }
	}
	if (verbose_) { cerr << " " << word_count->size() << " types" << endl; }
    }
    size_t num_words = 0;
    for (const auto &word_pair : *word_count) { num_words += word_pair.second; }
    return num_words;
}

size_t Corpus::BuildWordDictionary(const unordered_map<string, size_t> &count,
				   size_t rare_cutoff,
				   unordered_map<string, Word>
				   *word_dictionary) {
    size_t num_considered_words = 0;
    bool have_rare = false;
    word_dictionary->clear();
    for (const auto &string_count_pair : count) {
	string word_string = string_count_pair.first;
	size_t word_count = string_count_pair.second;
	if (word_count > rare_cutoff) {
	    num_considered_words += word_count;
	    (*word_dictionary)[word_string] = word_dictionary->size();
	} else {
	    have_rare = true;
	}
    }
    if (have_rare) {  // The rare word symbol gets the highest index.
	(*word_dictionary)[corpus::kRareString] = word_dictionary->size();
    }
    return num_considered_words;
}

size_t Corpus::SlideWindow(const unordered_map<string, Word> &word_dictionary,
			   bool sentence_per_line,
			   const string &context_definition, size_t window_size,
			   size_t hash_size,
			   unordered_map<string, Context> *context_dictionary,
			   unordered_map<Context, unordered_map<Word, double> >
			   *context_word_count) {
    Window window(window_size, context_definition, cooccur_weight_method_,
		  word_dictionary, corpus::kRareString, corpus::kBufferString,
		  hash_size, context_dictionary, context_word_count);

    // Preparation for random subsampling.
    random_device device;
    default_random_engine engine(device());
    uniform_real_distribution<double> uniform(0.0, 1.0);  // Uniform.
    double total_word_count = 0.0;  // For normalization.
    if (word_count_.size() > 0) {
	for (const auto &word_pair : word_count_) {
	    total_word_count += word_pair.second;
	}
    }
    double rare_probability = 0.0;
    if (word_count_.size() > 0 &&
	word_count_.find(corpus::kRareString) != word_count_.end()) {
	rare_probability = double(word_count_[corpus::kRareString])
	    / total_word_count;
    }

    vector<string> file_list;
    util_file::list_files(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = util_file::get_num_lines(file_path);
	if (verbose_) {
	    cerr << "Sliding window in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    util_file::read_line(&file, &word_strings);
	    ++line_num;
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    for (string word_string : word_strings) {
		if (Skip(word_string)) { continue; }
		if (lowercase_) {
		    word_string = util_string::lowercase(word_string);
		}
		if (subsampling_threshold_ > 0.0 && word_count_.size() > 0) {
		    double word_probability = (word_count_.find(word_string) !=
					       word_count_.end()) ?
			double(word_count_[word_string]) / total_word_count :
			rare_probability;
		    if (word_probability > subsampling_threshold_) {
			double random_probability = uniform(engine);
			double discard_probability =
			    1 - sqrt(subsampling_threshold_ / word_probability);
			if (discard_probability > random_probability) {
			    continue;  // Randomly skip frequent words.
			}
		    }
		}
		window.Add(word_string);
	    }
	    if (sentence_per_line) { window.Finish(); } // Finish the line.
	    if (line_num / num_lines >= portion_so_far) {
		portion_so_far += kReportInterval_;
		if (verbose_) { cerr << "." << flush; }
	    }
	}
	if (!sentence_per_line) { window.Finish(); } // Finish the file.
	if (verbose_) { cerr << endl; }
    }
    size_t num_nonzeros = 0;
    for (const auto &context_pair : *context_word_count) {
	num_nonzeros += context_pair.second.size();
    }
    return num_nonzeros;
}

void Corpus::CountTransitions(
    const unordered_map<string, Word> &word_dictionary,
    unordered_map<Word, unordered_map<Word, size_t> > *bigram_count,
    unordered_map<Word, size_t> *start_count,
    unordered_map<Word, size_t> *end_count) {
    bigram_count->clear();
    start_count->clear();
    end_count->clear();

    vector<string> file_list;
    util_file::list_files(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = util_file::get_num_lines(file_path);
	if (verbose_) {
	    cerr << "Counting transitions in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    util_file::read_line(&file, &word_strings);
	    ++line_num;
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    Word w_prev;
	    for (size_t i = 0; i < word_strings.size(); ++i) {
		string word_string = (!lowercase_) ? word_strings[i] :
		    util_string::lowercase(word_strings[i]);
		if (Skip(word_string)) { continue; }
		if (word_dictionary.find(word_string) ==
		    word_dictionary.end()) {
		    word_string = corpus::kRareString;
		}
		Word w = word_dictionary.at(word_string);

		if (i == 0) { ++(*start_count)[w]; }
		if (i == word_strings.size() - 1) { ++(*end_count)[w]; }
		if (i > 0) { ++(*bigram_count)[w_prev][w]; }
		w_prev = w;
	    }
	    if (line_num / num_lines >= portion_so_far) {
		portion_so_far += kReportInterval_;
		if (verbose_) { cerr << "." << flush; }
	    }
	}
	if (verbose_) { cerr << endl; }
    }
}

void Corpus::LoadWordCounts(const string &word_count_path, size_t rare_cutoff) {
    word_count_.clear();
    ifstream file(word_count_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << word_count_path);
    while (file.good()) {
	vector<string> tokens;
	util_file::read_line(&file, &tokens);
	if (tokens.size() > 0) {
	    size_t count = stol(tokens[1]);
	    if (count > rare_cutoff) {
		word_count_[tokens[0]] = count;
	    } else {
		word_count_[corpus::kRareString] += count;
	    }
	}
    }
}

bool Corpus::Skip(const string &word_string) {
    return (word_string == corpus::kRareString ||  // Special "rare" symbol.
	    word_string == corpus::kBufferString ||  // Special "buffer" symbol.
	    word_string.size() > kMaxWordLength_);  // Too long.
}

void Window::Add(const string &word_string) {
    // Filter words before putting in the window.
    string word_string_filtered = (word_dictionary_.find(word_string) !=
				   word_dictionary_.end()) ?
	word_string : rare_symbol_;
    queue_.push_back(word_string_filtered);  // [dog saw the]--[dog saw the cat]
    if (queue_.size() == window_size_) {
	ProcessFull();
	queue_.pop_front();  // [dog saw the cat]--[saw the cat]
    }
}

void Window::Finish() {
    size_t num_window_elements = queue_.size();
    while (queue_.size() < window_size_) {
	// This can happen if the number of words added to the window was
	// smaller than the window size (note that in this case the window was
	// never processed). So first fill up the window:
	queue_.push_back(buffer_symbol_);  // [<!> he did]--[<!> he did <!>]
    }
    for (size_t i = center_index_; i < num_window_elements; ++i) {
	ProcessFull();
	queue_.pop_front();  // [<!> he did <!>]--[he did <!>]
	queue_.push_back(buffer_symbol_);  // [he did <!>]--[he did <!> <!>]
    }
    queue_.clear();
    for (size_t i = 0; i < center_index_; ++i) {
	queue_.push_back(buffer_symbol_);
    }
}

void Window::PrepareWindow() {
    ASSERT(window_size_ >= 2, "Window size less than 2: " << window_size_);
    center_index_ = (window_size_ - 1) / 2;  // Right-biased center index.

    // Buffer the window up to before the center index.
    for (size_t i = 0; i < center_index_; ++i) {
	queue_.push_back(buffer_symbol_);
    }

    // Initialize the string markers for position-sensitive contexts.
    // Also initialize co-occurrence weights.
    position_markers_.resize(window_size_);
    cooccur_weights_.resize(window_size_);
    for (size_t i = 0; i < window_size_; ++i) {
	if (i != center_index_) {
	    int relative_position = int(i) - int(center_index_);
	    position_markers_[i] = "c(" + to_string(relative_position) + ")=";
	    if (cooccur_weight_method_ == "unif") {
		cooccur_weights_[i] = 1.0;
	    } else if (cooccur_weight_method_ == "inv") {
		cooccur_weights_[i] = 1.0 / fabs(relative_position);
	    } else {
		ASSERT(false, "Unknown cooccurrence weight method: "
		       << cooccur_weight_method_);
	    }
	}
    }
}

void Window::ProcessFull() {
    Word word = word_dictionary_.at(queue_[center_index_]);

    for (size_t i = 0; i < window_size_; ++i) {
	if (i == center_index_) { continue; }
	if (context_definition_ == "bag") {  // Bag-of-words contexts
	    Context bag_context = AddContextString(queue_[i]);
	    (*context_word_count_)[bag_context][word] += cooccur_weights_[i];
	} else if (context_definition_ == "list") {  // List-of-words contexts
	    Context list_context = AddContextString(position_markers_[i] +
						    queue_[i]);
	    (*context_word_count_)[list_context][word] += cooccur_weights_[i];
	} else {
	    ASSERT(false, "Unknown context definition: " <<
		   context_definition_);
	}
    }
}

Context Window::AddContextString(const string &context_string) {
    string context_string_hashed = (hash_size_ == 0) ?  // Context hashing
	context_string : to_string(context_hash_(context_string) % hash_size_);

    if (context_dictionary_->find(context_string_hashed) ==
	context_dictionary_->end()) {  // Add to dictionary if not known.
	Context new_context_index = context_dictionary_->size();
	(*context_dictionary_)[context_string_hashed] = new_context_index;
    }
    return (*context_dictionary_)[context_string_hashed];
}
