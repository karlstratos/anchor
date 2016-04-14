// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the utility code.

#include "gtest/gtest.h"

#include <algorithm>

#include "../util.h"

// Checks the string form of a printf format string.
TEST(PrintfFormatString, CheckBehavior) {
    string test_string = "TEST_STRING";
    float test_float = 3.14159;
    float test_float_carry = 3.1489;
    size_t test_long = 999999999999999;
    string string_string =
	util_string::printf_format("String: %s", test_string.c_str());
    string float_string =
	util_string::printf_format("Float: %.2f", test_float);
    string float_carry_string =
	util_string::printf_format("Float carry: %.2f", test_float_carry);
    string science_string =
	util_string::printf_format("Science: %.2e", test_float);
    string long_string =
	util_string::printf_format("Long: %ld", test_long);
    string percent_string =
	util_string::printf_format("Percent: 100%%");
    EXPECT_EQ("String: TEST_STRING", string_string);
    EXPECT_EQ("Float: 3.14", float_string);
    cout << test_float_carry << endl;
    cout << float_carry_string << endl;
    EXPECT_EQ("Float carry: 3.15", float_carry_string);
    EXPECT_EQ("Science: 3.14e+00", science_string);
    EXPECT_EQ("Long: 999999999999999", long_string);
    EXPECT_EQ("Percent: 100%", percent_string);
}

// Test class for string tokenization.
class StringTokenization : public testing::Test {
protected:
    virtual void SetUp() {
	example_ = "I have	some\n tabs	and spaces";
    }
    string example_;
};

// Checks spliting by a string delimiter.
TEST_F(StringTokenization, SplitByString) {
    vector<string> tokens_by_phrase;
    util_string::split_by_string(example_, "some\n tabs", &tokens_by_phrase);
    EXPECT_EQ(2, tokens_by_phrase.size());
    EXPECT_EQ("I have\t", tokens_by_phrase[0]);
    EXPECT_EQ("\tand spaces", tokens_by_phrase[1]);

    vector<string> tokens_by_space;
    util_string::split_by_string(example_, " ", &tokens_by_space);
    EXPECT_EQ(4, tokens_by_space.size());
    EXPECT_EQ("I", tokens_by_space[0]);
    EXPECT_EQ("have	some\n", tokens_by_space[1]);
    EXPECT_EQ("tabs	and", tokens_by_space[2]);
    EXPECT_EQ("spaces", tokens_by_space[3]);
}

// Checks spliting by char delimiters.
TEST_F(StringTokenization, SplitByChars) {
    vector<string> tokens_by_whitespace;
    util_string::split_by_chars(example_, " \t\n", &tokens_by_whitespace);
    EXPECT_EQ(6, tokens_by_whitespace.size());
    EXPECT_EQ("I", tokens_by_whitespace[0]);
    EXPECT_EQ("have", tokens_by_whitespace[1]);
    EXPECT_EQ("some", tokens_by_whitespace[2]);
    EXPECT_EQ("tabs", tokens_by_whitespace[3]);
    EXPECT_EQ("and", tokens_by_whitespace[4]);
    EXPECT_EQ("spaces", tokens_by_whitespace[5]);
}

// Checks reading lines from a text file.
TEST(UtilString, FileNextLineTokenization) {
    string text_file_path = tmpnam(nullptr);
    ofstream text_file_out(text_file_path, ios::out);
    text_file_out << "a b	c" << endl;
    text_file_out << endl;
    text_file_out << "		d e f" << endl;
    text_file_out << endl;
    text_file_out.close();

    ifstream text_file_in(text_file_path, ios::in);
    vector<string> tokens;

    //  "a b\tc"
    util_file::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("a", tokens[0]);
    EXPECT_EQ("b", tokens[1]);
    EXPECT_EQ("c", tokens[2]);

    //  ""
    util_file::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  "\t\td e f"
    util_file::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("d", tokens[0]);
    EXPECT_EQ("e", tokens[1]);
    EXPECT_EQ("f", tokens[2]);

    //  ""
    util_file::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  ""
    util_file::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    remove(text_file_path.c_str());
}

// Checks converting seconds to string.
TEST(UtilString, ConvertSecondsToString) {
    EXPECT_EQ("20h7m18s", util_string::convert_seconds_to_string(72438.1));
    EXPECT_EQ("20h7m18s", util_string::convert_seconds_to_string(72438.9));
}

// Checks lowercasing a string.
TEST(UtilString, Lowercase) {
    EXPECT_EQ("ab12345cd@#%! ?ef", util_string::lowercase("AB12345Cd@#%! ?eF"));
}

// Checks converting a vector to string.
TEST(UtilString, ConvertVectorToString) {
    EXPECT_EQ("a b c", util_string::convert_to_string({"a", "b", "c"}));
    EXPECT_EQ("1 2 0.54", util_string::convert_to_string({1.0, 2.0, 0.538}));
    EXPECT_EQ("0.53", util_string::convert_to_string({0.532}));
    EXPECT_EQ("1 2 3", util_string::convert_to_string({1, 2, 3}));
}

// Checks checking the existence of a file.
TEST(UtilFile, FileExists) {
    string file_path = tmpnam(nullptr);
    ofstream file_out(file_path, ios::out);
    file_out.close();
    EXPECT_TRUE(util_file::exists(file_path));
    remove(file_path.c_str());
    EXPECT_FALSE(util_file::exists(file_path));
}

// Test class for file writing/reading.
class FileWritingReading : public testing::Test {
protected:
    virtual void SetUp() {
	flat_[0.31] = 7;
	double_nested_[0][-3] = 0.0;
	double_nested_[100][-2] = 1.0 / 3.0;
	flat_string_to_size_t_["The"] = 0;
	flat_string_to_size_t_["elephant"] = 1;
	flat_string_to_size_t_["saw"] = 2;
	flat_string_to_size_t_["."] = 3;
    }
    unordered_map<float, size_t> flat_;
    unordered_map<size_t, unordered_map<int, double> > double_nested_;
    unordered_map<string, size_t> flat_string_to_size_t_;
    double tol_ = 1e-6;
};

// Checks writing/reading a flat unordered_map of primitive types.
TEST_F(FileWritingReading, FlatUnorderedMapPrimitive) {
    string file_path = tmpnam(nullptr);
    util_file::binary_write_primitive(flat_, file_path);

    unordered_map<float, size_t> table;
    util_file::binary_read_primitive(file_path, &table);
    EXPECT_EQ(1, table.size());
    EXPECT_EQ(7, table[0.31]);
    remove(file_path.c_str());
}

// Checks writing/reading a 2-nested unordered_map of primitive types.
TEST_F(FileWritingReading, DoubleNestedUnorderedMapPrimitive) {
    string file_path = tmpnam(nullptr);
    util_file::binary_write_primitive(double_nested_, file_path);

    unordered_map<size_t, unordered_map<int, double> > table;
    util_file::binary_read_primitive(file_path, &table);
    EXPECT_EQ(2, table.size());
    EXPECT_EQ(1, table[0].size());
    EXPECT_NEAR(0.0, table[0][-3], tol_);
    EXPECT_EQ(1, table[100].size());
    EXPECT_NEAR(1.0 / 3.0, table[100][-2], tol_);
    remove(file_path.c_str());
}

// Checks writing/reading a flat (string, size_t) unordered_map.
TEST_F(FileWritingReading, FlatUnorderedMapStringSizeT) {
    string file_path = tmpnam(nullptr);
    util_file::binary_write(flat_string_to_size_t_, file_path);

    unordered_map<string, size_t> table;
    util_file::binary_read(file_path, &table);
    EXPECT_EQ(4, table.size());
    EXPECT_EQ(0, table["The"]);
    EXPECT_EQ(1, table["elephant"]);
    EXPECT_EQ(2, table["saw"]);
    EXPECT_EQ(3, table["."]);
    remove(file_path.c_str());
}

// Checks the average-rank transform of a sequence.
TEST(UtilMath, TransformAverageRank) {
    vector<double> sequence =   {3,  -5,   4,   1,   1,  9,  10,  10};
    //         Sorted:          {-5,  1,   1,   3,   4,  9,  10,  10}
    //         Ranks:           <1,   2,   3,   4,   5,  6,   7,   8>
    //         Average ranks:   <1, 2.5, 2.5,   4,   5,  6, 7.5, 7.5>
    //         Unsorted:        <4,   1,   5, 2.5, 2.5,  6, 7.5, 7.5>
    vector<double> average_ranks;
    util_math::transform_average_rank(sequence, &average_ranks);

    double tol = 1e-10;
    EXPECT_EQ(8, average_ranks.size());
    EXPECT_NEAR(4.0, average_ranks[0], tol);
    EXPECT_NEAR(1.0, average_ranks[1], tol);
    EXPECT_NEAR(5.0, average_ranks[2], tol);
    EXPECT_NEAR(2.5, average_ranks[3], tol);
    EXPECT_NEAR(2.5, average_ranks[4], tol);
    EXPECT_NEAR(6.0, average_ranks[5], tol);
    EXPECT_NEAR(7.5, average_ranks[6], tol);
    EXPECT_NEAR(7.5, average_ranks[7], tol);
}

// Checks sorting a vector of pairs by the second values.
TEST(UtilMisc, SortVectorOfPairsBySecondValues) {
    double tol = 1e-6;
    vector<pair<string, double> > pairs;
    pairs.emplace_back("a", 3.0);
    pairs.emplace_back("b", 0.09);
    pairs.emplace_back("c", 100);

    // Sort in increasing magnitude.
    sort(pairs.begin(), pairs.end(),
	 util_misc::sort_pairs_second<string, double>());
    EXPECT_EQ("b", pairs[0].first);
    EXPECT_NEAR(0.09, pairs[0].second, tol);
    EXPECT_EQ("a", pairs[1].first);
    EXPECT_NEAR(3.0, pairs[1].second, tol);
    EXPECT_EQ("c", pairs[2].first);
    EXPECT_NEAR(100.0, pairs[2].second, tol);

    // Sort in decreasing magnitude.
    sort(pairs.begin(), pairs.end(),
	 util_misc::sort_pairs_second<string, double, greater<int> >());
    EXPECT_EQ("c", pairs[0].first);
    EXPECT_NEAR(100.0, pairs[0].second, tol);
    EXPECT_EQ("a", pairs[1].first);
    EXPECT_NEAR(3.0, pairs[1].second, tol);
    EXPECT_EQ("b", pairs[2].first);
    EXPECT_NEAR(0.09, pairs[2].second, tol);
}

// Checks subtracting by the median value in unordered_map.
TEST(UtilMisc, SubtractByMedian) {
    unordered_map<string, size_t> table;
    table["a"] = 100;
    table["b"] = 80;
    table["c"] = 5;
    table["d"] = 3;
    table["e"] = 3;
    table["f"] = 3;
    table["g"] = 1;
    table["h"] = 1;
    table["i"] = 1;
    table["j"] = 1;
    // 100 80 5 3 3 3 1 1 1 1
    //  a  b  c d e f g h i j
    //  0  1  2 3 4 5 6 7 8 9
    //            ^
    //          median
    util_misc::subtract_by_median(&table);

    // Should have a:97, b:77, and c:2 left.
    EXPECT_EQ(3, table.size());
    EXPECT_EQ(97, table["a"]);
    EXPECT_EQ(77, table["b"]);
    EXPECT_EQ(2, table["c"]);
}

// Checks inverting an unordered_map.
TEST(UtilMisc, InvertUnorderedMap) {
    unordered_map<string, size_t> table1;
    table1["a"] = 0;
    table1["b"] = 1;
    unordered_map<size_t, string> table2;
    util_misc::invert(table1, &table2);
    EXPECT_EQ(2, table2.size());
    EXPECT_EQ("a", table2[0]);
    EXPECT_EQ("b", table2[1]);
}

// Checks summing values in an unordered map.
TEST(UtilMisc, SumValuesInFlatUnorderedMap) {
    unordered_map<string, size_t> table;
    table["a"] = 7;
    table["b"] = 3;
    EXPECT_EQ(10, util_misc::sum_values(table));
}

// Checks summing values in a 2-nested unordered map.
TEST(UtilMisc, SumValuesInDoubleNestedUnorderedMap) {
    unordered_map<double, unordered_map<string, int> > table;
    table[0.75]["a"] = -7;
    table[0.75]["b"] = -5;
    table[0.31]["a"] = -3;
    EXPECT_EQ(-15, util_misc::sum_values(table));
}

// Checks if two unordered maps are near.
TEST(UtilMisc, CheckNearFlatUnorderedMaps) {
    unordered_map<string, double> table1;
    unordered_map<string, double> table2;
    table1["a"] = 7;
    table1["b"] = 7.1;
    table2["a"] = 7;
    table2["b"] = 7.1;
    EXPECT_TRUE(util_misc::check_near(table1, table2));

    table1["c"] = 7.000000001;
    table2["c"] = 7.000000002;
    EXPECT_FALSE(util_misc::check_near(table1, table2));
}

// Checks if two 2-nested unordered maps are near.
TEST(UtilMisc, CheckNearDoubleNestedUnorderedMaps) {
    unordered_map<string, unordered_map<string, double> > table1;
    unordered_map<string, unordered_map<string, double> > table2;
    table1["a"]["b"] = 7;
    table2["a"]["b"] = 7;
    EXPECT_TRUE(util_misc::check_near(table1, table2));

    table1["a"]["c"] = 7.000000001;
    table2["a"]["c"] = 7.000000002;
    EXPECT_FALSE(util_misc::check_near(table1, table2));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
