// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for optimization.

#ifndef CORE_OPTIMIZE_H_
#define CORE_OPTIMIZE_H_

#include <Eigen/Dense>

#include "util.h"

namespace optimize {
    // Finds the convex coefficients for the columns of a matrix that minimize
    // the squared loss wrt. to a target vector.
    void compute_convex_coefficients_squared_loss(
	const Eigen::MatrixXd &columns, const Eigen::VectorXd &target_vector,
	size_t max_num_updates, double stopping_threshold, bool verbose,
	Eigen::VectorXd *convex_coefficients);

    // Given rows that form a convex hull, finds vertex rows.
    void find_vertex_rows(const Eigen::MatrixXd &rows, size_t num_vertices,
			  const unordered_map<size_t, bool> &vertex_candidates,
			  vector<size_t> *vertex_indices);

    // Given vertex rows, extract each row of A as convex coefficients
    // (M=AB, see below).
    void extract_matrix(const Eigen::MatrixXd &M, size_t rank,
			size_t max_num_updates, double stopping_threshold,
			bool verbose, const vector<size_t>
			&vertex_indices, Eigen::MatrixXd *A);

    // Performs a special case of nonnegative matrix factorization (NMF) based
    // on the combinatorial method of Arora et al. (2012). Recovers A given
    // M=AB (d x d') where A (d x m) and B (m x d') are rank-m matrices.
    // Assumes the following structure on A:
    //
    //    1. A.row(i) is a distribution [p(1|i) ... p(m|i)]. Thus M.row(i) is a
    //       convex combination of the rows of B:
    //
    //          M.row(i) = A.row(i) * B.transpose()
    //
    //    2. For h=1...m, there is some index anchor(h) in {1...d} such that:
    //
    //          A.row(anchor(h)) = [0 0 ... 1 ... 0 0]
    //                                      ^
    //                                     h-th
    //
    //       (Thus M.row(anchor(h)) = B.row(h)!)
    //
    // Assumption 2 implies that M.row(i) is therefore some convex combination
    // of M.row(anchor(1)) ... M.row(anchor(m)), and the convex coefficients are
    // exactly A.row(i).
    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, const unordered_map<size_t, bool>
			      &anchor_candidates, vector<size_t>
			      *anchor_indices, Eigen::MatrixXd *A);

    // Anchor factorization with anchor indices given.
    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, const vector<size_t>
			      &anchor_indices, Eigen::MatrixXd *A);

    // Anchor factorization without anchor candidate restriction.
    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, vector<size_t> *anchor_indices,
			      Eigen::MatrixXd *A);
}  // namespace optimize

#endif  // CORE_OPTIMIZE_H_
