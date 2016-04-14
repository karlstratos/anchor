// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "features.h"

#include <assert.h>

namespace features {
    string basic_word_shape(const string &word_string) {
	bool is_all_digit = true;
	bool is_all_uppercase = true;
	bool is_all_lowercase = true;
	bool is_capitalized = true;
	int index = 0;
	for (const char &c : word_string) {
	    if (!isdigit(c)) is_all_digit = false;
	    if (!isupper(c)) is_all_uppercase = false;
	    if (!islower(c)) is_all_lowercase = false;
	    if ((index == 0 && !isupper(c)) || (index > 0 && !islower(c))) {
		is_capitalized = false;
	    }
	    ++index;
	}

	if (is_all_digit) {
	    return "all-digit";
	} else if (is_all_uppercase) {
	    return "all-upper";
	} else if (is_all_lowercase) {
	    return "all-lower";
	} else if (is_capitalized) {
	    return "capitalized";
	} else {
	    return "other";
	}
    }

    string word_identity(const string &word_string) { return word_string; }

    string prefix(const string &word_string, size_t prefix_length) {
	assert(prefix_length <= word_string.size());
	return word_string.substr(0, prefix_length);
    }

    string suffix(const string &word_string, size_t suffix_length) {
	assert(suffix_length <= word_string.size());
	return word_string.substr(word_string.size() - suffix_length);
    }

    string contains_digit(const string &word_string) {
	for (char c : word_string) {
	    if (isdigit(c)) { return "t"; }
	}
	return "f";
    }

    string contains_hyphen(const string &word_string) {
	for (char c : word_string) {
	    if (c == 45) { return "t"; }
	}
	return "f";
    }

    string is_capitalized(const string &word_string) {
	return isupper(word_string.at(0)) ? "t" : "f";
    }
}
