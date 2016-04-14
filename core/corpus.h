// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for processing text corpora.

#ifndef CORE_CORPUS_H_
#define CORE_CORPUS_H_

#include <Eigen/Dense>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sparsesvd.h"
#include "util.h"

typedef size_t Word;
typedef size_t Context;

namespace corpus {
    // Special string for representing rare words.
    const string kRareString = "<?>";

    // Special string for representing the out-of-sentence buffer.
    const string kBufferString = "<!>";

    // Performs a spectral decomposition of a transformed and scaled
    // word-context co-occurrence matrix. Some useful examples:
    //                             Transform          Add       Power     Scale
    //     -------------------------------------------------------------------
    //     SVD/LSA              {power, log}            0    {0.5, 1}      none
    //     PPMI-SVD                    power            0           1      ppmi
    //     (Ridge) Regression          power          200           1       reg
    //     Regression + sqrt           power            0         0.5       reg
    //     (Regularized) CCA           power          200           1       cca
    //     CCA + sqrt                  power            0         0.5       cca
    void decompose(SMat matrix, size_t desired_rank,
		   const string &transformation_method, double add_smooth,
		   double power_smooth, double context_power_smooth,
		   const string &scaling_method,
		   Eigen::MatrixXd *left_singular_vectors,
		   Eigen::MatrixXd *right_singular_vectors,
		   Eigen::VectorXd *singular_values);

    // No power smoothing for context distributions.
    void decompose(SMat matrix, size_t desired_rank,
		   const string &transformation_method, double add_smooth,
		   double power_smooth, const string &scaling_method,
		   Eigen::MatrixXd *left_singular_vectors,
		   Eigen::MatrixXd *right_singular_vectors,
		   Eigen::VectorXd *singular_values);

    // Transforms the given (count) value.
    double transform(double count_value, double add_smooth,
		     double power_smooth, string transformation_method);

    // Loads the word-count pairs > cutoff (in decreasing frequency).
    void load_sorted_word_types(size_t rare_cutoff,
				const string &sorted_word_types_path,
				vector<pair<string, size_t> >
				*sorted_word_types);

    // Loads word vectors: [count] [string] [dim 1] ... [dim m].
    void load_word_vectors(const string &word_vectors_path,
			   unordered_map<string, Eigen::VectorXd>
			   *word_vectors, bool normalize);

    // Loads sorted word vectors from a file with a pre-sorted vector per line.
    void load_sorted_word_vectors(const string &sorted_word_vectors_path,
				  vector<size_t> *sorted_word_counts,
				  vector<string> *sorted_word_strings,
				  vector<Eigen::VectorXd> *sorted_word_vectors,
				  bool normalize);
}  // namespace corpus

// A Corpus object is designed to extract useful statistics about text easily.
class Corpus {
public:
    // Always initializes with a corpus.
    Corpus(const string &corpus_path, bool verbose) : corpus_path_(corpus_path),
						      verbose_(verbose) { }

    ~Corpus() { }

    // Writes word types sorted in decreasing frequency as a text file. Also
    // writes a word dictionary as a binary file.
    void WriteWords(size_t rare_cutoff, const string &sorted_word_types_path,
		    const string &word_dictionary_path, size_t *num_words,
		    size_t *num_word_types, size_t *vocabulary_size);

    // Writes a context dictionary as a binary file. Also writes the
    // co-occurrence counts between words and contexts as an SVDLIBC binary
    // file.
    void WriteContexts(const unordered_map<string, Word> &word_dictionary,
		       bool sentence_per_line, const string &context_definition,
		       size_t window_size, size_t hash_size,
		       const string &context_dictionary_path,
		       const string &context_word_count_path,
		       size_t *num_nonzeros);

    // Writes word transition counts as binary files.
    void WriteTransitions(const unordered_map<string, Word> &word_dictionary,
			  const string &bigram_count_path,
			  const string &start_count_path,
			  const string &end_count_path);

    // Counts words appearing in the corpus, returns the total count.
    size_t CountWords(unordered_map<string, size_t> *word_count);

    // Builds a dictionary of word types from their counts. Those with counts <=
    // rare_cutoff are considered a single "rare word" (which by convention gets
    // the highest index). Returns the total count of word types considered in
    // the dictionary.
    size_t BuildWordDictionary(const unordered_map<string, size_t> &word_count,
			       size_t rare_cutoff,
			       unordered_map<string, Word> *word_dictionary);

    // Slides a window across the corpus (within a word dictionary):
    //    - sentence_per_line:  Each line of the corpus is a sentence?
    //    - context_definition: How "context" is defined.
    //    - window_size:        Size of the sliding window (odd => symmetric,
    //                          even => assymmetric).
    //    - hash_size:          Number of hash bins (0 means no hashing).
    // Returns the number of nonzeros in the co-occurrence map.
    size_t SlideWindow(const unordered_map<string, Word> &word_dictionary,
		       bool sentence_per_line,
		       const string &context_definition, size_t window_size,
		       size_t hash_size,
		       unordered_map<string, Context> *context_dictionary,
		       unordered_map<Context, unordered_map<Word, double> >
		       *context_word_count);

    // Counts word transitions (within a word dictionary):
    //    - bigram_count[w1][w2]: count of bigram (w1, w2)
    //    - start_count[w]:       count of word w starting a sentence
    //    - end_count[w]:         count of word w ending a sentence
    // Note that the corpus must have a sentence per line for start_count and
    // end_count to be meaningful.
    void CountTransitions(const unordered_map<string, Word> &word_dictionary,
			  unordered_map<Word, unordered_map<Word, size_t> >
			  *bigram_count, unordered_map<Word, size_t>
			  *start_count, unordered_map<Word, size_t> *end_count);

    // Load word counts from a text file with lines: <word_string> <word_count>.
    void LoadWordCounts(const string &word_count_path, size_t rare_cutoff);

    // Sets the flag for lowercasing all strings.
    void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

    // Sets the maximum vocabulary size.
    void set_max_vocabulary_size(size_t max_vocabulary_size) {
	max_vocabulary_size_ = max_vocabulary_size;
    }

    // Sets the subsampling threshold.
    void set_subsampling_threshold(double subsampling_threshold) {
	subsampling_threshold_ = subsampling_threshold;
    }

    // Sets the co-occurrence weight method.
    void set_cooccur_weight_method(string cooccur_weight_method) {
	cooccur_weight_method_ = cooccur_weight_method;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Returns the corpus path.
    string corpus_path() { return corpus_path_; }

private:
    // Returns true if the string is to be skipped.
    bool Skip(const string &word_string);

    // Maximum sentence length to consider.
    const size_t kMaxSentenceLength_ = 1000;

    // Maximum word length to consider.
    const size_t kMaxWordLength_ = 100;

    // Interval to report progress (0.1 => every 10%).
    const double kReportInterval_ = 0.1;

    // Path to a corpus (a single text file, or a directory of text files).
    string corpus_path_;

    // Lowercase all strings?
    bool lowercase_ = false;

    // Maximum vocabulary size allowed.
    size_t max_vocabulary_size_ = 1000000000;  // 1 billion

    // Internal word counts (only loaded when necessary).
    unordered_map<string, size_t> word_count_;

    // Subsampling threshold.
    double subsampling_threshold_ = 0.0;

    // Co-occurrence weight method.
    string cooccur_weight_method_ = "unif";

    // Print messages to stderr?
    bool verbose_ = true;
};

// A Window object is used to "slide" a window of fixed size and count
// word-context co-occurrences. The definition of a "context" is customized
// (bag-of-words, list, etc.) and a context dictionary is built on the fly.
// Optionally, contexts can be randomly hashed to control their number.
//
// E.g., Bag-of-words contexts, window size 4, sentence "the dog saw the cat":
//        window            call                count([word], [context])
//   [<!>]
//   [<!> the]           Add("the")
//   [<!> the dog]       Add("dog")
//   [<!> the dog saw]   Add("saw")       ++(the, <!>) ++(the, dog) ++(the, saw)
//   [the dog saw the]   Add("the")       ++(dog, the) ++(dog, saw) ++(dog, the)
//   [dog saw the cat]   Add("cat")       ++(saw, dog) ++(saw, the) ++(saw, cat)
//   ______________________________
//   [saw the cat <!>]   Finish()         ++(the, saw) ++(the, cat) ++(the, <!>)
//   [the cat <!> <!>]                    ++(cat, the) ++(cat, <!>) ++(cat, <!>)
//   [<!>]
class Window {
public:
    // Initializes a window.
    Window(size_t window_size, const string &context_definition,
	   string cooccur_weight_method,
	   const unordered_map<string, Word> &word_dictionary,
	   const string &rare_symbol, const string &buffer_symbol,
	   size_t hash_size, unordered_map<string, Context> *context_dictionary,
	   unordered_map<Context, unordered_map<Word, double> >
	   *context_word_count) :
	window_size_(window_size), context_definition_(context_definition),
	cooccur_weight_method_(cooccur_weight_method),
	word_dictionary_(word_dictionary), rare_symbol_(rare_symbol),
	buffer_symbol_(buffer_symbol), hash_size_(hash_size),
	context_dictionary_(context_dictionary),
	context_word_count_(context_word_count) { PrepareWindow(); }

    // Adds a word string to the window and processes it if full.
    void Add(const string &word_string);

    // Processes whatever is in the window and resets.
    void Finish();

private:
    // Prepares a window with given parameters.
    void PrepareWindow();

    // Increments co-occurrence counts from a full window of text.
    void ProcessFull();

    // Adds the given string to the context dictionary (if not already in it).
    // Returns a unique index corresponding to this string.
    Context AddContextString(const string &context_string);

    // FIFO queue of strings.
    deque<string> queue_;

    // Center index of the window.
    size_t center_index_;

    // Strings for marking position-sensitive contexts.
    vector<string> position_markers_;

    // Co-occurrence weights.
    vector<double> cooccur_weights_;

    // Hash function for hashing context strings.
    hash<string> context_hash_;

    // Window size.
    size_t window_size_;

    // Context definition.
    string context_definition_;

    // Co-occurrence weight method.
    string cooccur_weight_method_ = "unif";

    // Address of the word dictionary.
    const unordered_map<string, Word> &word_dictionary_;

    // Rare symbol.
    string rare_symbol_;

    // Buffer symbol.
    string buffer_symbol_;

    // Number of hash bins (0 means no hashing).
    size_t hash_size_;

    // Pointer to a context dictionary.
    unordered_map<string, Context> *context_dictionary_;

    // Pointer to a context-word co-occurrence counts.
    unordered_map<Context, unordered_map<Word, double> > *context_word_count_;
};

#endif  // CORE_CORPUS_H_
