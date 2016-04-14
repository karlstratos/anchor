// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "sparsesvd.h"

#include <fstream>
#include <iomanip>
#include <math.h>
#include <sstream>

namespace sparsesvd {
    SMat binary_read_sparse_matrix(const string &file_path) {
	ifstream file(file_path, ios::in | ios::binary);
	size_t num_rows;
	size_t num_columns;
	size_t num_nonzeros;
	util_file::binary_read_primitive(file, &num_rows);
	util_file::binary_read_primitive(file, &num_columns);
	util_file::binary_read_primitive(file, &num_nonzeros);

	// Load the sparse matrix variable.
	SMat sparse_matrix = svdNewSMat(num_rows, num_columns, num_nonzeros);

	size_t current_nonzero_index = 0;  // Keep track of nonzero values.
	for (size_t col = 0; col < num_columns; ++col) {
	    sparse_matrix->pointr[col] = current_nonzero_index;
	    size_t num_nonzero_rows;
	    util_file::binary_read_primitive(file, &num_nonzero_rows);
	    for (size_t i = 0; i < num_nonzero_rows; ++i) {
		size_t row;
		double value;  // Value guaranteed to be double.
		util_file::binary_read_primitive(file, &row);
		util_file::binary_read_primitive(file, &value);
		sparse_matrix->rowind[current_nonzero_index] = row;
		sparse_matrix->value[current_nonzero_index] = value;
		++current_nonzero_index;
	    }
	}
	sparse_matrix->pointr[num_columns] = num_nonzeros;
	return sparse_matrix;
    }

    void compute_svd(SMat sparse_matrix, size_t desired_rank,
		     Eigen::MatrixXd *left_singular_vectors,
		     Eigen::MatrixXd *right_singular_vectors,
		     Eigen::VectorXd *singular_values, size_t *actual_rank) {
	if (desired_rank == 0) {
	    left_singular_vectors->resize(0, 0);
	    right_singular_vectors->resize(0, 0);
	    singular_values->resize(0);
	    (*actual_rank) = 0;
	    return;
	}
	size_t rank_upper_bound = min(sparse_matrix->rows, sparse_matrix->cols);
	if (desired_rank > rank_upper_bound) {  // Adjust the oversized rank.
	    desired_rank = rank_upper_bound;
	}

	// Run the Lanczos algorithm with default parameters.
	SVDRec svd_result = svdLAS2A(sparse_matrix, desired_rank);

	(*actual_rank) = svd_result->d;  // This is the actual SVD rank.

	singular_values->resize(*actual_rank);
	for (size_t i = 0; i < *actual_rank; ++i) {
	    (*singular_values)(i) = *(svd_result->S + i);
	}

	left_singular_vectors->resize(sparse_matrix->rows, *actual_rank);
	for (size_t row = 0; row < sparse_matrix->rows; ++row) {
	    for (size_t col = 0; col < *actual_rank; ++col) {
		(*left_singular_vectors)(row, col) =
		    svd_result->Ut->value[col][row];  // Transpose.
	    }
	}

	right_singular_vectors->resize(sparse_matrix->cols, *actual_rank);
	for (size_t row = 0; row < sparse_matrix->cols; ++row) {
	    for (size_t col = 0; col < *actual_rank; ++col) {
		(*right_singular_vectors)(row, col) =
		    svd_result->Vt->value[col][row];  // Transpose.
	    }
	}

	svdFreeSVDRec(svd_result);
    }

    double sum_rows_columns(SMat sparse_matrix,
			    unordered_map<size_t, double> *row_sum,
			    unordered_map<size_t, double> *column_sum) {
	row_sum->clear();
	column_sum->clear();
	double total_sum = 0.0;
	size_t current_nonzero_index = 0;
	for (size_t col = 0; col < sparse_matrix->cols; ++col) {
	    while (current_nonzero_index < sparse_matrix->pointr[col + 1]) {
		size_t row = sparse_matrix->rowind[current_nonzero_index];
		double value = sparse_matrix->value[current_nonzero_index];
		(*row_sum)[row] += value;
		(*column_sum)[col] += value;
		total_sum += value;
		++current_nonzero_index;
	    }
	}
	return total_sum;
    }
}  // namespace sparsesvd
