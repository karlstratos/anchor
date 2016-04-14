// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the optimization code.

#include "gtest/gtest.h"

#include <limits>

#include "../optimize.h"

// Test class for convex-constrained optimization.
class ConvexConstrainedOptimization : public testing::Test {
protected:
    // (1 x 3): short and fat
    // (3 x 3): square
    // (6 x 3): tall and thin
    vector<size_t> list_num_rows_ = {1, 3, 6};
    vector<size_t> list_num_columns_ = {3};
    size_t max_num_updates_ = numeric_limits<size_t>::max();
    double stopping_threshold_ = 1e-10;
    bool verbose_ = false;
};

// Checks minimizing the squared-loss for convex-constrained optimization.
TEST_F(ConvexConstrainedOptimization, SquaredLoss) {
    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    // Set up a problem.
	    Eigen::MatrixXd columns = Eigen::MatrixXd::Random(num_rows,
							      num_columns);
	    Eigen::VectorXd convex_coefficients =
		Eigen::VectorXd::Random(num_columns).cwiseAbs();
	    double l1_norm = convex_coefficients.lpNorm<1>();
	    convex_coefficients /= l1_norm;
	    Eigen::VectorXd target_vector = columns * convex_coefficients;

	    // Solve the problem.
	    Eigen::VectorXd computed_coefficients;
	    optimize::compute_convex_coefficients_squared_loss(
		columns, target_vector, max_num_updates_, stopping_threshold_,
		verbose_, &computed_coefficients);
	    Eigen::VectorXd estimate = columns * computed_coefficients;

	    // Check if the error (= loss of the estimate since the optimal loss
	    // is 0) is at least as small as specified.
	    double error = 0.5 * (target_vector - estimate).squaredNorm();
	    EXPECT_TRUE(error <= stopping_threshold_);
	}
    }
}

// Test class for anchor factorization.
class AnchorFactorization : public testing::Test {
protected:
    virtual void SetUp() {
	// Each row of A is a distribution...
	A_ = Eigen::MatrixXd::Random(num_rows_, rank_).cwiseAbs();
	for (size_t row = 0; row < A_.rows(); ++row) {
	    A_.row(row) /= A_.row(row).lpNorm<1>();
	}
	// ... plus, A satisfies the anchor assumption. Each column i has the
	// corresponding anchor row: i for simplicity.
	for (size_t col = 0; col < A_.cols(); ++col) {
	    A_.row(col) = Eigen::VectorXd::Zero(A_.cols());
	    A_.row(col)(col) = 1.0;
	}

	// B is any full-rank matrix.
	B_ = Eigen::MatrixXd::Random(rank_, num_columns_);
	M_ = A_ * B_;
    }

    virtual void TearDown() { }

    size_t num_rows_ = 6;
    size_t num_columns_ = 8;
    size_t rank_ = 3;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;
    Eigen::MatrixXd M_;
    size_t max_num_updates_ = numeric_limits<size_t>::max();
    double stopping_threshold_ = 1e-10;
    bool verbose_ = false;
    double tol_ = 1e-5;
};

// Checks if the anchor factorization algorithm correctly recovers the A matrix.
TEST_F(AnchorFactorization, CorrectlyRecoverA) {
    vector<size_t> anchor_indices;
    Eigen::MatrixXd recovered_A;
    optimize::anchor_factorization(M_, rank_, max_num_updates_,
				   stopping_threshold_, verbose_,
				   &anchor_indices, &recovered_A);
    for (size_t row = 0; row < A_.rows(); ++row) {
	for (size_t column = 0; column < A_.cols(); ++column) {
	    EXPECT_NEAR(A_(row, column), recovered_A(row, column), tol_);
	}
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
