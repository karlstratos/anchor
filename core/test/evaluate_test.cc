// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the evaluation code.

#include "gtest/gtest.h"

#include <fstream>

#include "../evaluate.h"

// Test class that provides example sequences.
class SequenceExample : public testing::Test {
protected:
    vector<vector<string> > true_sequences_ = {{"a", "b", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences_matching_labels_ =
    {{"a", "c", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences_unmatching_labels_ =
    {{"0", "1", "2"}, {"1", "2"}};
    double position_accuracy_;
    double sequence_accuracy_;
};


// Checks evaluating sequence predictions with matching labels.
TEST_F(SequenceExample, MatchingLabels) {
    eval_sequential::compute_accuracy(true_sequences_,
				      predicted_sequences_matching_labels_,
				      &position_accuracy_, &sequence_accuracy_);
    EXPECT_NEAR(80.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(50.0, sequence_accuracy_, 1e-15);
}

// Checks evaluating sequence predictions with unmatching labels.
TEST_F(SequenceExample, UnmatchingLabels) {
    eval_sequential::compute_accuracy(true_sequences_,
			       predicted_sequences_unmatching_labels_,
			       &position_accuracy_, &sequence_accuracy_);
    EXPECT_NEAR(0.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(0.0, sequence_accuracy_, 1e-15);

    unordered_map<string, string> label_mapping;
    eval_sequential::compute_accuracy_mapping_labels(
	true_sequences_, predicted_sequences_unmatching_labels_,
	&position_accuracy_, &sequence_accuracy_, &label_mapping);
    EXPECT_NEAR(100.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(100.0, sequence_accuracy_, 1e-15);

    EXPECT_EQ("a", label_mapping["0"]);
    EXPECT_EQ("b", label_mapping["1"]);
    EXPECT_EQ("c", label_mapping["2"]);
}

// Checks evaluting word similarity with a random example.
TEST(WordSimilarity, RandomExample) {
    Eigen::VectorXd x1(2);
    Eigen::VectorXd x2(2);
    Eigen::VectorXd x3(2);
    Eigen::VectorXd x4(2);
    Eigen::VectorXd x5(2);
    Eigen::VectorXd x6(2);
    Eigen::VectorXd y1(2);
    Eigen::VectorXd y2(2);
    Eigen::VectorXd y3(2);
    Eigen::VectorXd y4(2);
    Eigen::VectorXd y5(2);
    Eigen::VectorXd y6(2);
    x1 << 0.9134, 0.6324;
    x2 << 0.0975, 0.2785;
    x3 << 0.5469, 0.9575;
    x4 << 0.1419, 0.4218;
    x5 << 0.9157, 0.7922;
    x6 << 0.9595, 0.6557;
    y1 << 0.9649, 0.1576;
    y2 << 0.9706, 0.9572;
    y3 << 0.4854, 0.8003;
    y4 << 0.0357, 0.8491;
    y5 << 0.9340, 0.6787;
    y6 << 0.7577, 0.7431;
    unordered_map<string, Eigen::VectorXd> word_vectors;
    word_vectors["x1"] = x1;
    word_vectors["x2"] = x2;
    word_vectors["x3"] = x3;
    word_vectors["x4"] = x4;
    word_vectors["x5"] = x5;
    word_vectors["x6"] = x6;
    word_vectors["y1"] = y1;
    word_vectors["y2"] = y2;
    word_vectors["y3"] = y3;
    word_vectors["y4"] = y4;
    word_vectors["y5"] = y5;
    word_vectors["y6"] = y6;
    string scored_word_pairs_path = tmpnam(nullptr);
    ofstream scored_word_pairs_file(scored_word_pairs_path, ios::out);
    scored_word_pairs_file << "x1 y1 -1" << endl;
    scored_word_pairs_file << "x2 y2 100" << endl;
    scored_word_pairs_file << "x3 y3 101" << endl;
    scored_word_pairs_file << "x4 y4 -3" << endl;
    scored_word_pairs_file << "x5 y5 1000" << endl;
    scored_word_pairs_file << "x6 y6 1002" << endl;
    scored_word_pairs_file << "x7 y1 1002" << endl;  // x7 not in dictionary.
    size_t num_instances;
    size_t num_handled;
    double correlation;
    eval_lexical::compute_correlation(scored_word_pairs_path, word_vectors,
				      false, &num_instances, &num_handled,
				      &correlation);
    EXPECT_EQ(7, num_instances);
    EXPECT_EQ(6, num_handled);
    EXPECT_NEAR(0.5429, correlation, 1e-4);

    remove(scored_word_pairs_path.c_str());
}

// Checks evaluting word analogy with a manual example.
TEST(WordAnalogy, ManualExample) {
    Eigen::VectorXd england(2);
    Eigen::VectorXd london(2);
    Eigen::VectorXd island(2);
    Eigen::VectorXd Korea(2);
    Eigen::VectorXd korea(2);
    Eigen::VectorXd Seoul(2);
    Eigen::VectorXd peninsula(2);
    Eigen::VectorXd play(2);
    Eigen::VectorXd played(2);
    Eigen::VectorXd playing(2);
    Eigen::VectorXd run(2);
    Eigen::VectorXd ran(2);
    Eigen::VectorXd running(2);
    //                                                          |
    //                                                          |
    //                                       peninsula         Korea       Seoul
    //                                                          |
    //                                       island          england      london
    //                                                          |
    // -------------------------------------------------------------------------
    //                                                          |
    //                                                          |
    //                                                          |
    //                                                          |
    //                                                          |
    //                played       play      playing            |
    //                                                          |
    //                  ran        run      running             |
    //                                                          |
    //                            korea                         |
    england << 0, 1;
    london << 1, 1;
    island << -1, 1;
    Korea << 0, 2;
    korea << -100, -102;
    Seoul << 1, 2;
    peninsula << -1, 2;
    play << -100, -100;
    played << -101, -100;
    playing << -99, -100;
    run << -100, -101;
    ran << -101, -101;
    running << -99, -101;
    unordered_map<string, Eigen::VectorXd> word_vectors;
    word_vectors["england"] = england;
    word_vectors["london"] = london;
    word_vectors["island"] = island;
    word_vectors["Korea"] = Korea;
    word_vectors["korea"] = korea;
    word_vectors["Seoul"] = Seoul;
    word_vectors["peninsula"] = peninsula;
    word_vectors["play"] = play;
    word_vectors["played"] = played;
    word_vectors["playing"] = playing;
    word_vectors["run"] = run;
    word_vectors["ran"] = ran;
    word_vectors["running"] = running;
    string analogy_questions_path = tmpnam(nullptr);
    ofstream analogy_questions_file(analogy_questions_path, ios::out);
    analogy_questions_file << "semantic England London Korea Seoul" << endl;
    analogy_questions_file << "semantic England island Korea peninsula" << endl;
    analogy_questions_file << "semantic w1 w2 v1 v2" << endl;
    analogy_questions_file << "syntactic play played run ran" << endl;
    analogy_questions_file << "syntactic play playing run running" << endl;
    analogy_questions_file << "syntactic w1 w2 v1 v2" << endl;
    size_t num_instances;
    size_t num_handled;
    double accuracy;
    unordered_map<string, double> per_type_accuracy;
    eval_lexical::compute_analogy_accuracy(analogy_questions_path, word_vectors,
					   false, &num_instances, &num_handled,
					   &accuracy, &per_type_accuracy);
    EXPECT_EQ(6, num_instances);
    EXPECT_EQ(4, num_handled);
    EXPECT_NEAR(100.0, accuracy, 1e-10);
    EXPECT_NEAR(100.0, per_type_accuracy["semantic"], 1e-10);
    EXPECT_NEAR(100.0, per_type_accuracy["syntactic"], 1e-10);

    word_vectors.erase("Korea");  // Now "korea" is used, throwing off semantic.
    eval_lexical::compute_analogy_accuracy(analogy_questions_path, word_vectors,
					   false, &num_instances, &num_handled,
					   &accuracy, &per_type_accuracy);
    EXPECT_EQ(6, num_instances);
    EXPECT_EQ(4, num_handled);
    EXPECT_NEAR(50.0, accuracy, 1e-10);
    EXPECT_NEAR(0.0, per_type_accuracy["semantic"], 1e-10);
    EXPECT_NEAR(100.0, per_type_accuracy["syntactic"], 1e-10);

    remove(analogy_questions_path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
