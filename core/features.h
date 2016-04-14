// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code related to feature extraction.

#ifndef CORE_FEATURES_H_
#define CORE_FEATURES_H_

#include <string>

using namespace std;

namespace features {
    // Returns the 5-way word shape feature.
    string basic_word_shape(const string &word_string);

    // Returns the word identity feature.
    string word_identity(const string &word_string);

    // Returns the word prefix feature.
    string prefix(const string &word_string, size_t prefix_length);

    // Returns the word suffix feature.
    string suffix(const string &word_string, size_t suffix_length);

    // Boolean feature: Is there a digit in the word?
    string contains_digit(const string &word_string);

    // Boolean feature: Is there a hyphen ("-") in the word?
    string contains_hyphen(const string &word_string);

    // Boolean feature: Is the word capitalized?
    string is_capitalized(const string &word_string);
}  // namespace features

#endif  // CORE_FEATURES_H_
