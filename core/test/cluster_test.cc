// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the clustering code.

#include "gtest/gtest.h"

#include <limits>

#include "../cluster.h"

// Test class that provides a set of vectors for clustering.
class VectorsForClustering : public testing::Test {
protected:
    virtual void SetUp() {
	Eigen::VectorXd v0(1);
	Eigen::VectorXd v1(1);
	Eigen::VectorXd v2(1);
	Eigen::VectorXd v3(1);
	Eigen::VectorXd v4(1);
	Eigen::VectorXd v5(1);
	v0 << 0.0;
	v3 << 0.3;
	v1 << 3.0;
	v4 << 3.9;
	v2 << 9.0;
	v5 << 9.6;
	//       0 0.3           3     3.9                           9  9.6
	//------v0-v3------------v1----v4----------------------------v2--v5-----
	ordered_vectors_.push_back(v0);
	ordered_vectors_.push_back(v1);
	ordered_vectors_.push_back(v2);
	ordered_vectors_.push_back(v3);
	ordered_vectors_.push_back(v4);
	ordered_vectors_.push_back(v5);
	// Structures are disambiguated by the "left < right" children ordering.
	// All 6 leaves:
	//                                 /\
	//                                /  \
	//                               /    \
	//                              /\    / \
	//                            v2 v5  /   \
	//                                  /\   /\
	//                                v0 v3 v1 v4
	all_leaves_paths_ = {"100", "110", "00", "101", "111", "01"};
	// 2 leaves without pruning:
	//                                 /\
	//                                /  \
	//                               /\   \
	//                             v4 /\   \
	//                              v3 /\   /\
	//                               v0 v1 v2 v5
	two_leaves_paths_ = {"0110", "0111", "10", "010", "00", "11"};
	// 2 leaves with pruning:
	//                                   /\
	//                      {v0,v1,v3,v4} {v2,v5}
	two_leaves_pruned_paths_ = {"0", "0", "1", "0", "0", "1"};
    }
    vector<Eigen::VectorXd> ordered_vectors_;
    AgglomerativeClustering agglomerative_;
    vector<string> all_leaves_paths_;
    vector<string> two_leaves_paths_;
    vector<string> two_leaves_pruned_paths_;
};

// Checks agglomerative clustering with all 6 leaves.
TEST_F(VectorsForClustering, AgglomerativeAll6Leaves) {
    size_t num_leaf_clusters = numeric_limits<size_t>::max();
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < all_leaves_paths_.size(); ++i) {
	EXPECT_EQ(all_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering twice.
TEST_F(VectorsForClustering, AgglomerativeAll6LeavesTwice) {
    size_t num_leaf_clusters = numeric_limits<size_t>::max();
    double gamma1 = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							 num_leaf_clusters);
    // Cluster again.
    double gamma2 = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							 num_leaf_clusters);
    EXPECT_NEAR(gamma1, gamma2, 1e-15);
    for (size_t i = 0; i < all_leaves_paths_.size(); ++i) {
	EXPECT_EQ(all_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering with 2 leaf clusters without pruning.
TEST_F(VectorsForClustering, Agglomerative2LeavesNotPruned) {
    size_t num_leaf_clusters = 2;
    agglomerative_.set_prune(false);  // Do not prune.
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < two_leaves_pruned_paths_.size(); ++i) {
	EXPECT_EQ(two_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering with 2 leaf clusters with pruning.
TEST_F(VectorsForClustering, Agglomerative2LeavesPruned) {
    size_t num_leaf_clusters = 2;
    agglomerative_.set_prune(true);  // Prune.
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < two_leaves_pruned_paths_.size(); ++i) {
	EXPECT_EQ(two_leaves_pruned_paths_[i],
		  agglomerative_.path_from_root(i));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
