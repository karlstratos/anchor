// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various helper functions for the Eigen library. Some conventions:
//    - A "basis" is always a matrix whose columns are the basis elements.

#ifndef CORE_EIGEN_HELPER_H_
#define CORE_EIGEN_HELPER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "util.h"

namespace eigen_helper {
    // Converts an unordered map (column -> {row: value}) to an Eigen sparse
    // matrix.
    template<typename T, typename EigenSparseMatrix>
    void convert_column_map(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map,
	EigenSparseMatrix *matrix) {
	matrix->resize(0, 0);
	size_t num_rows = 0;
	size_t num_columns = 0;
	vector<Eigen::Triplet<T> > triplet_list;  // {(row, column, value)}
	size_t num_nonzeros = 0;
	for (const auto &column_pair: column_map) {
	    num_nonzeros += column_pair.second.size();
	}
	triplet_list.reserve(num_nonzeros);
	for (const auto &column_pair: column_map) {
	    size_t column = column_pair.first;
	    if (column >= num_columns) { num_columns = column + 1; }
	    for (const auto &row_pair: column_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
		T value = row_pair.second;
		triplet_list.emplace_back(row, column, value);
	    }
	}
	matrix->resize(num_rows, num_columns);
	matrix->setFromTriplets(triplet_list.begin(), triplet_list.end());
    }

    // Writes an Eigen dense matrix to a binary file.
    template<typename EigenDenseMatrix>
    void binary_write_matrix(const EigenDenseMatrix& matrix,
			     const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenDenseMatrix::Index num_rows = matrix.rows();
	typename EigenDenseMatrix::Index num_columns = matrix.cols();
	util_file::binary_write_primitive(num_rows, file);
	util_file::binary_write_primitive(num_columns, file);
	file.write(reinterpret_cast<const char *>(matrix.data()), num_rows *
		   num_columns * sizeof(typename EigenDenseMatrix::Scalar));
    }

    // Reads an Eigen dense matrix from a binary file.
    template<typename EigenDenseMatrix>
    void binary_read_matrix(const string &file_path, EigenDenseMatrix *matrix) {
	ifstream file(file_path, ios::in | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenDenseMatrix::Index num_rows;
	typename EigenDenseMatrix::Index num_columns;
	util_file::binary_read_primitive(file, &num_rows);
	util_file::binary_read_primitive(file, &num_columns);
	matrix->resize(num_rows, num_columns);
	file.read(reinterpret_cast<char*>(matrix->data()), num_rows *
		  num_columns * sizeof(typename EigenDenseMatrix::Scalar));
    }

    // Computes the Moore–Penrose pseudo-inverse.
    void compute_pseudoinverse(const Eigen::MatrixXd &matrix,
			       Eigen::MatrixXd *matrix_pseudoinverse);

    // Extends an orthonormal basis to subsume the given vector v.
    void extend_orthonormal_basis(const Eigen::VectorXd &v,
				  Eigen::MatrixXd *orthonormal_basis);

    // Finds an orthonormal basis that spans the range of the matrix.
    void find_range(const Eigen::MatrixXd &matrix,
		    Eigen::MatrixXd *orthonormal_basis);

    // Computes principal component analysis (PCA). The format of the input
    // matrix is: rows = samples, columns = dimensions.
    void compute_pca(const Eigen::MatrixXd &original_sample_rows,
		     Eigen::MatrixXd *rotated_sample_rows,
		     Eigen::MatrixXd *rotation_matrix,
		     Eigen::VectorXd *variances);

    // Generates a random projection matrix.
    void generate_random_projection(size_t original_dimension,
				    size_t reduced_dimension,
				    Eigen::MatrixXd *projection_matrix);

    // Returns true if two Eigen dense matrices are close in value.
    template<typename EigenDenseMatrix>
    bool check_near(const EigenDenseMatrix& matrix1,
		    const EigenDenseMatrix& matrix2, double error_threshold) {
	if (matrix1.rows() != matrix2.rows() ||
	    matrix2.cols() != matrix2.cols()) { return false; }
	for (size_t row = 0; row < matrix1.rows(); ++row) {
	    for (size_t col = 0; col < matrix1.cols(); ++col) {
		if (fabs(matrix1(row, col) - matrix2(row, col))
		    > error_threshold) { return false; }
	    }
	}
	return true;
    }

    // Returns true if two Eigen dense matrices are close in absolute value.
    template<typename EigenDenseMatrix>
    bool check_near_abs(const EigenDenseMatrix& matrix1,
			const EigenDenseMatrix& matrix2,
			double error_threshold) {
	if (matrix1.rows() != matrix2.rows() ||
	    matrix2.cols() != matrix2.cols()) { return false; }
	for (size_t row = 0; row < matrix1.rows(); ++row) {
	    for (size_t col = 0; col < matrix1.cols(); ++col) {
		if (fabs(fabs(matrix1(row, col)) - fabs(matrix2(row, col)))
		    > error_threshold) { return false; }
	    }
	}
	return true;
    }

    // Computes the KL divergence of distribution 2 from distribution 1.
    // WARNING: Assign distribution variables before passing them, e.g.,
    //          don't do "kl_divergence(M.col(0), M.col(1));
    template<typename EigenDenseVector>
    double kl_divergence(const EigenDenseVector& distribution1,
			 const EigenDenseVector& distribution2) {
	double kl = 0.0;
	for (size_t i = 0; i < distribution1.size(); ++i) {
	    if (distribution2(i) <= 0.0) {
		ASSERT(distribution1(i) <= 0.0, "KL is undefined");
	    }
	    if (distribution1(i) > 0.0) {
		kl += distribution1(i) * (log(distribution1(i)) -
					  log(distribution2(i)));
	    }
	}
	return kl;
    }
}  // namespace eigen_helper

#endif  // CORE_EIGEN_HELPER_H_
