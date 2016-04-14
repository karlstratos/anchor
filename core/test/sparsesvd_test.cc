// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the sparse SVD code.

#include "gtest/gtest.h"
#include "../sparsesvd.h"

// Checks writing/reading an int column map as a file gives the correct SMat.
TEST(ObtainingSMat, WriteReadIntMatrixAsFile) {
    double tol = 1e-10;
    unordered_map<size_t, unordered_map<size_t, int> > column_map;
    column_map[2][0] = -3;
    column_map[0][1] = 2;
    string file_path = tmpnam(nullptr);
    // 0 0 -3
    // 2 0  0
    sparsesvd::binary_write_sparse_matrix(column_map, file_path);
    SMat sparse_matrix = sparsesvd::binary_read_sparse_matrix(file_path);
    EXPECT_EQ(2, sparse_matrix->rows);
    EXPECT_EQ(3, sparse_matrix->cols);
    size_t current_nonzero_index = 0;
    for (size_t col = 0; col < sparse_matrix->cols; ++col) {
	while (current_nonzero_index < sparse_matrix->pointr[col + 1]) {
	    EXPECT_NE(1, col);  // Column 1 is empty.
	    size_t row = sparse_matrix->rowind[current_nonzero_index];
	    double value = sparse_matrix->value[current_nonzero_index];
	    if (col == 0) {
		EXPECT_EQ(1, row);
		EXPECT_NEAR(2.0, value, tol);
	    }
	    if (col == 2) {
		EXPECT_EQ(0, row);
		EXPECT_NEAR(-3.0, value, tol);
	    }
	    ++current_nonzero_index;
	}
    }
    svdFreeSMat(sparse_matrix);
    remove(file_path.c_str());
}

// Checks converting a double column map gives the correct SMat.
TEST(ObtainingSMat, ConvertDoubleColumnMap) {
    double tol = 1e-10;
    unordered_map<size_t, unordered_map<size_t, double> > column_map;
    column_map[1][2] = 3.14159;
    // 0       0
    // 0       0
    // 0 3.14159
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);
    EXPECT_EQ(3, sparse_matrix->rows);
    EXPECT_EQ(2, sparse_matrix->cols);
    size_t current_nonzero_index = 0;
    for (size_t col = 0; col < sparse_matrix->cols; ++col) {
	while (current_nonzero_index < sparse_matrix->pointr[col + 1]) {
	    EXPECT_NE(0, col);  // Column 0 is empty.
	    size_t row = sparse_matrix->rowind[current_nonzero_index];
	    double value = sparse_matrix->value[current_nonzero_index];
	    EXPECT_EQ(2, row);
	    EXPECT_NEAR(3.14159, value, tol);
	    ++current_nonzero_index;
	}
    }
    svdFreeSMat(sparse_matrix);
}

// Checks computing SVD of a full-rank matrix.
TEST(ComputingSVD, FullRankMatrix) {
    double tol = 1e-10;
    size_t num_rows = 4;
    size_t num_columns = 5;
    size_t desired_rank = num_rows + num_columns;  // Oversized rank.
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(num_rows, num_columns);
    unordered_map<size_t, unordered_map<size_t, double> > column_map;
    sparsesvd::convert_eigen_dense_to_column_map(matrix, &column_map);
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);  // "sparse"

    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    size_t actual_rank;
    sparsesvd::compute_svd(sparse_matrix, desired_rank, &left_singular_vectors,
			   &right_singular_vectors, &singular_values,
			   &actual_rank);

    EXPECT_EQ(min(num_rows, num_columns), actual_rank);  // Full-rank.
    Eigen::MatrixXd reconstructed_matrix = left_singular_vectors *
	singular_values.asDiagonal() * right_singular_vectors.transpose();
    EXPECT_NEAR(0.0, (matrix - reconstructed_matrix).norm(), tol);  // Frobenius
    svdFreeSMat(sparse_matrix);
}

// Checks computing SVD of a low-rank matrix.
TEST(ComputingSVD, LowRankMatrix) {
    double tol = 1e-10;
    size_t num_rows = 8;
    size_t num_columns = 16;
    size_t low_rank = 3;
    size_t desired_rank = num_rows + num_columns;  // Oversized rank.
    ASSERT(low_rank < num_rows && low_rank < num_columns, "Set low rank");

    // First construct some random low-rank matrix.
    Eigen::MatrixXd left_matrix = Eigen::MatrixXd::Random(num_rows, low_rank);
    Eigen::MatrixXd right_matrix = Eigen::MatrixXd::Random(num_columns,
							   low_rank);
    Eigen::MatrixXd matrix = left_matrix * right_matrix.transpose();
    unordered_map<size_t, unordered_map<size_t, double> > column_map;
    sparsesvd::convert_eigen_dense_to_column_map(matrix, &column_map);
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);  // "sparse"

    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    size_t actual_rank;
    sparsesvd::compute_svd(sparse_matrix, desired_rank, &left_singular_vectors,
			   &right_singular_vectors, &singular_values,
			   &actual_rank);

    EXPECT_EQ(low_rank, actual_rank);  // Low rank.
    EXPECT_EQ(low_rank, singular_values.size());
    EXPECT_EQ(num_rows, left_singular_vectors.rows());
    EXPECT_EQ(low_rank, left_singular_vectors.cols());
    EXPECT_EQ(num_columns, right_singular_vectors.rows());
    EXPECT_EQ(low_rank, right_singular_vectors.cols());
    Eigen::MatrixXd reconstructed_matrix = left_singular_vectors *
	singular_values.asDiagonal() * right_singular_vectors.transpose();
    EXPECT_NEAR(0.0, (matrix - reconstructed_matrix).norm(), tol);  // Frobenius
    svdFreeSMat(sparse_matrix);
}

// This test demonstrates that the SVD computed by SVDLIBC is *incorrect* when
// the gap between the largest two singular values is truly (close to) zero.
// This is achieved by an identity matrix.
TEST(UnstableWithSmallSingularGap, IncorrectWithIdentity) {
    size_t dimension = 2;
    size_t desired_rank = dimension;
    unordered_map<size_t, unordered_map<size_t, size_t> > column_map;
    for (size_t i = 0; i < dimension; ++i) { column_map[i][i] = 1; }
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);

    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    size_t actual_rank;
    sparsesvd::compute_svd(sparse_matrix, desired_rank, &left_singular_vectors,
			   &right_singular_vectors, &singular_values,
			   &actual_rank);

    EXPECT_NE(dimension, actual_rank);  // Should have been the dimension!
    svdFreeSMat(sparse_matrix);
}

// This test demonstrates that when the gap between the largest two singular
// values is small but not zero, the SVD computed by SVDLIBC is correct.
TEST(UnstableWithSmallSingularGap, CorrectWithSomeSingularGap) {
    double tol = 1e-10;

    // The following is an artificially constructed matrix of rank 2 that has
    // singular values 2 and 1.99998.
    unordered_map<size_t, unordered_map<size_t, double> > column_map;
    column_map[0][0] = -0.0992;
    column_map[0][1] = 1.7317;
    column_map[1][0] = 1.9287;
    column_map[1][1] = -0.1764;
    column_map[2][0] = 0.5198;
    column_map[2][1] = 0.9850;
    size_t desired_rank = 2;
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);

    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    size_t actual_rank;
    sparsesvd::compute_svd(sparse_matrix, desired_rank, &left_singular_vectors,
			   &right_singular_vectors, &singular_values,
			   &actual_rank);

    EXPECT_EQ(2, actual_rank);  // Correctly 2!
    Eigen::MatrixXd matrix;
    sparsesvd::convert_column_map_to_eigen_dense(column_map, &matrix);
    Eigen::MatrixXd reconstructed_matrix = left_singular_vectors *
	singular_values.asDiagonal() * right_singular_vectors.transpose();
    EXPECT_NEAR(0.0, (matrix - reconstructed_matrix).norm(), tol);  // Frobenius
    svdFreeSMat(sparse_matrix);
}

// Checks summing rows/columns.
TEST(SumRowsColumns, Correctness) {
    double tol = 1e-10;
    //     0  0  1  1
    //     0  0  0  1
    //     2  0  3  1
    //     0  0  0  1
    unordered_map<size_t, unordered_map<size_t, size_t> > column_map;
    column_map[0][2] = 2;
    column_map[2][0] = 1;
    column_map[2][2] = 3;
    column_map[3][0] = 1;
    column_map[3][1] = 1;
    column_map[3][2] = 1;
    column_map[3][3] = 1;
    SMat sparse_matrix = sparsesvd::convert_column_map(column_map);
    unordered_map<size_t, double> row_sum;
    unordered_map<size_t, double> column_sum;
    double total_sum =
	sparsesvd::sum_rows_columns(sparse_matrix, &row_sum, &column_sum);

    EXPECT_NEAR(2.0, row_sum[0], tol);
    EXPECT_NEAR(1.0, row_sum[1], tol);
    EXPECT_NEAR(6.0, row_sum[2], tol);
    EXPECT_NEAR(1.0, row_sum[3], tol);
    EXPECT_NEAR(2.0, column_sum[0], tol);
    EXPECT_NEAR(0.0, column_sum[1], tol);
    EXPECT_NEAR(4.0, column_sum[2], tol);
    EXPECT_NEAR(4.0, column_sum[3], tol);
    EXPECT_NEAR(10.0, total_sum, tol);
    svdFreeSMat(sparse_matrix);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
