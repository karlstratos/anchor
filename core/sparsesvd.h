// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// This is a wrapper around SVDLIBC that provides a clean interface for
// performing singular value decomposition (SVD) on sparse matrices in
// Standard C++. The general workflow is as follows:
//
//                      [Write/read SMat file](*)
//                      /                 \
// [Build a column map]                     [Compute SVD] - [Free SMat]
//                      \                 /
//                       [Convert to SMat]
//
// (*) Recommended for large matrices.
// Always free the memory allocated to SMat objects after using them. E.g.,
//    SMat sparse_matrix = binary_read_sparse_matrix(sparse_matrix_path);
//    ...
//    svdFreeSMat(sparse_matrix);  // Make sure to do this!

#ifndef CORE_SPARSESVD_H_
#define CORE_SPARSESVD_H_

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

#include "util.h"

extern "C" {  // For using C code from C++ code.
#include "../third_party/SVDLIBC/svdlib.h"
}

namespace sparsesvd {
    // Writes an unordered_map (column -> {row: value}) as a binary file for
    // SVDLIBC.
    template<typename T>
    void binary_write_sparse_matrix(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map,
	const string &file_path) {
	size_t num_rows = 0;
	size_t num_columns = 0;
	size_t num_nonzeros = 0;
	for (const auto &col_pair: column_map) {
	    size_t col = col_pair.first;
	    if (col >= num_columns) { num_columns = col + 1; }
	    for (const auto &row_pair: col_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
		++num_nonzeros;
	    }
	}

	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	util_file::binary_write_primitive(num_rows, file);
	util_file::binary_write_primitive(num_columns, file);
	util_file::binary_write_primitive(num_nonzeros, file);
	for (size_t col = 0; col < num_columns; ++col) {
	    if (column_map.find(col) == column_map.end()) {
		size_t zero = 0;  // No nonzero rows for this column.
		util_file::binary_write_primitive(zero, file);
		continue;
	    }
	    util_file::binary_write_primitive(column_map.at(col).size(), file);
	    for (const auto &row_pair: column_map.at(col)) {
		size_t row = row_pair.first;
		double value = double(row_pair.second);  // Convert to double.
		util_file::binary_write_primitive(row, file);
		util_file::binary_write_primitive(value, file);
	    }
	}
    }

    // Reads an SVDLIBC sparse matrix from a binary file.
    SMat binary_read_sparse_matrix(const string &file_path);

    // Computes a low-rank SVD of a sparse matrix in the SVDLIBC format. The
    // singular vectors are organized as columns of a matrix. The given matrix
    // has rank smaller than the desired rank if actual_rank < desired_rank:
    // in this case, the SVD result is automatically truncated.
    void compute_svd(SMat sparse_matrix, size_t desired_rank,
		     Eigen::MatrixXd *left_singular_vectors,
		     Eigen::MatrixXd *right_singular_vectors,
		     Eigen::VectorXd *singular_values, size_t *actual_rank);

    // Sums the rows/columns of a sparse matrix M: row_sum[i] = sum_j M_{i, j},
    // column_sum[j] = sum_i M_{i, j}. Returns the total sum.
    double sum_rows_columns(SMat sparse_matrix,
			    unordered_map<size_t, double> *row_sum,
			    unordered_map<size_t, double> *column_sum);

    // Converts a column map to an SVDLIBC sparse matrix M:
    //    M_{i, j} = column_map[j][i]
    // Not recommended for large matrices since at least twice the memory is
    // needed: in that case, load an SMat object directly from a file.
    template<typename T>
    SMat convert_column_map(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map) {
	size_t num_rows = 0;
	size_t num_columns = 0;
	size_t num_nonzeros = 0;
	for (const auto &col_pair: column_map) {
	    size_t col = col_pair.first;
	    if (col >= num_columns) { num_columns = col + 1; }
	    for (const auto &row_pair: col_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
		++num_nonzeros;
	    }
	}

	// Load the sparse matrix variable.
	SMat sparse_matrix = svdNewSMat(num_rows, num_columns, num_nonzeros);

	size_t current_nonzero_index = 0;  // Keep track of nonzero values.
	for (size_t col = 0; col < num_columns; ++col) {
	    sparse_matrix->pointr[col] = current_nonzero_index;
	    if (column_map.find(col) == column_map.end()) { continue; }
	    for (const auto &row_pair: column_map.at(col)) {
		size_t row = row_pair.first;
		double value = double(row_pair.second);  // Convert to double.
		sparse_matrix->rowind[current_nonzero_index] = row;
		sparse_matrix->value[current_nonzero_index] = value;
		++current_nonzero_index;
	    }
	}
	sparse_matrix->pointr[num_columns] = num_nonzeros;
	return sparse_matrix;
    }

    // Converts an Eigen dense matrix to a column map.
    template<typename EigenDenseMatrix>
    void convert_eigen_dense_to_column_map(
	const EigenDenseMatrix &matrix,
	unordered_map<size_t, unordered_map<size_t, double> > *column_map) {
	column_map->clear();
	for (size_t row = 0; row < matrix.rows(); ++row) {
	    for (size_t col = 0; col < matrix.cols(); ++col) {
		(*column_map)[col][row] = matrix(row, col);
	    }
	}
    }

    // Converts a column map to an Eigen dense matrix.
    template<typename EigenDenseMatrix>
    void convert_column_map_to_eigen_dense(
	const unordered_map<size_t, unordered_map<size_t, double> > &column_map,
	EigenDenseMatrix *matrix) {
	size_t num_rows = 0;
	size_t num_columns = 0;
	for (const auto &col_pair: column_map) {
	    size_t col = col_pair.first;
	    if (col >= num_columns) { num_columns = col + 1; }
	    for (const auto &row_pair: col_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
	    }
	}

	matrix->resize(num_rows, num_columns);
	for (size_t row = 0; row < num_rows; ++row) {
	    for (size_t col = 0; col < num_columns; ++col) {
		(*matrix)(row, col) = column_map.at(col).at(row);
	    }
	}
    }

    // Converts an Eigen dense matrix to an SVDLIBC sparse matrix M.
    template<typename EigenDenseMatrix>
    SMat convert_eigen_dense(const EigenDenseMatrix &matrix) {
	unordered_map<size_t, unordered_map<size_t, double> > column_map;
	convert_eigen_dense_to_column_map(matrix, &column_map);
	return convert_column_map(column_map);
    }
}  // namespace sparsesvd

#endif  // CORE_SPARSESVD_H_
