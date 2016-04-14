// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "optimize.h"

namespace optimize {
    void compute_convex_coefficients_squared_loss(
	const Eigen::MatrixXd &columns, const Eigen::VectorXd &target_vector,
	size_t max_num_updates, double stopping_threshold, bool verbose,
	Eigen::VectorXd *convex_coefficients) {
	ASSERT(target_vector.size() == columns.rows(), "Dimensions mismatch");
	size_t num_columns = columns.cols();
	convex_coefficients->resize(num_columns);
	if (num_columns == 0) { return; }  // Avoid the degenerate case.
	for (size_t i = 0; i < num_columns; ++i) {  // Uniform initialization.
	    (*convex_coefficients)(i) = 1.0 / num_columns;
	}

	// Iterate the Frank-Wolfe steps:
	for (size_t update_num = 1; update_num <= max_num_updates;
	     ++update_num) {
	    Eigen::VectorXd current_vector = columns * (*convex_coefficients);
	    Eigen::VectorXd residual = current_vector - target_vector;

	    // Step 1. Minimize the linear approximation function around the
	    // current solution inside the probability simplex.
	    Eigen::VectorXd gradient = columns.transpose() * residual;
	    double min_value = numeric_limits<double>::infinity();
	    size_t min_i = 0;
	    for (size_t i = 0; i < num_columns; ++i) {
		if (gradient(i) < min_value) {
		    min_value = gradient(i);
		    min_i = i;
		}
	    }

	    // If the duality gap (an upper bound on the difference between the
	    // current loss and the optimal loss) is small, stop optimizing.
	    if (update_num % 100 == 0) {
		Eigen::VectorXd deflated_vector = *convex_coefficients;
		deflated_vector(min_i) -= 1.0;
		double duality_gap = gradient.dot(deflated_vector);
		if (verbose) {
		    cerr << "\r" << update_num << ": CURRENT_LOSS - "
			 << "OPTIMAL_LOSS <= " << duality_gap << "   " << flush;
		}
		if (duality_gap <= stopping_threshold) { break; }
	    }

	    // Step 2. Find the optimal step size.
	    Eigen::VectorXd vector1 = current_vector - columns.col(min_i);
	    double vector1_norm = vector1.squaredNorm();
	    double step_size = vector1.dot(residual);  // 0 if vector1 = 0
	    if (vector1_norm > 0.0) {
		step_size /= vector1_norm;
		step_size = min(step_size, 1.0);
		step_size = max(step_size, 0.0);
	    }
	    if (std::isnan(step_size)) {  // Numerical underflow/overflow?
		step_size = 0.0;
	    }

	    // Step 3. Move the current x along coordinate min_i by step_size.
	    (*convex_coefficients) = (1 - step_size) * (*convex_coefficients);
	    (*convex_coefficients)(min_i) += step_size;
	}

	// Check if x is a proper distribution.
	bool is_negative = false;
	double l1_mass = 0.0;
	for (size_t i = 0; i < num_columns; ++i) {
	    double ith_prob = (*convex_coefficients)(i);
	    if (ith_prob < -1e-10) { is_negative = true; }
	    l1_mass += (*convex_coefficients)(i);
	}
	if (is_negative || fabs(l1_mass - 1.0) > 1e-10) {
	    cerr << endl << "WARNING: Computed improper distribution, "
		 << "will revert to uniform!!" << endl;
	    cerr << convex_coefficients->transpose() << endl;
	    return;
	}
    }

    void find_vertex_rows(const Eigen::MatrixXd &rows, size_t num_vertices,
			  const unordered_map<size_t, bool> &vertex_candidates,
			  vector<size_t> *vertex_indices) {
	vertex_indices->clear();

	// U is the orthonormal basis of the subspace spanned by currently
	// selected vertex rows.
	Eigen::MatrixXd U;
	unordered_map<size_t, bool> selected_rows;
	while (U.cols() < num_vertices) {
	    // Find the row furthest away from the current subspace.
	    Eigen::MatrixXd orthogonal_projection = U * U.transpose();
	    double max_distance = 0.0;
	    size_t vertex_row = 0;
	    for (size_t row = 0; row < rows.rows(); ++row) {
		// (RESTRICTION 1) Do not consider rows that are not candidates.
		if (vertex_candidates.find(row) ==
		    vertex_candidates.end()) { continue; }

		// (RESTRICTION 2) Do not consider rows already selected.
		if (selected_rows.find(row) !=
		    selected_rows.end()) { continue; }

		Eigen::VectorXd distance_vector = rows.row(row);
		if (U.cols() != 0) {  // Remove the subspace projection.
		    distance_vector -= orthogonal_projection * distance_vector;
		}
		double distance = distance_vector.norm();
		if (distance > max_distance) {
		    max_distance = distance;
		    vertex_row = row;
		}
	    }
	    selected_rows[vertex_row] = true;
	    vertex_indices->push_back(vertex_row);
	    Eigen::VectorXd new_direction = rows.row(vertex_row);
	    if (U.cols() != 0) {  // Remove the subspace projection.
		new_direction -= orthogonal_projection * new_direction;
	    }
	    new_direction.normalize();

	    // Extend the basis.
	    size_t vertex_num = U.cols() + 1;
	    U.conservativeResize(rows.cols(), vertex_num);
	    U.col(vertex_num - 1) = new_direction;
	}
	// Sort anchor indices by convention: e.g., [3 100 116 ... 2225].
	sort(vertex_indices->begin(), vertex_indices->end());
    }

    void extract_matrix(const Eigen::MatrixXd &M, size_t rank,
			size_t max_num_updates, double stopping_threshold,
			bool verbose, const vector<size_t>
			&vertex_indices, Eigen::MatrixXd *A) {
	// Organize vertex rows as columns of a matrix.
	Eigen::MatrixXd vertex_columns(M.cols(), rank);
	for (size_t i = 0; i < rank; ++i) {
	    vertex_columns.col(i) = M.row(vertex_indices.at(i));
	}

	// Recover each row of A as the convex coefficients for expressing
	// that row as a combination of vertex rows.
	A->resize(M.rows(), rank);
	for (size_t row = 0; row < A->rows(); ++row) {
	    Eigen::VectorXd convex_combination = M.row(row);
	    Eigen::VectorXd convex_coefficients;
	    compute_convex_coefficients_squared_loss(
		vertex_columns, convex_combination, max_num_updates,
		stopping_threshold, verbose, &convex_coefficients);
	    (*A).row(row) = convex_coefficients;
	    if (verbose) { cerr << row + 1 << "/" << A->rows() << "    "; }
	}
	if (verbose) { cerr << endl; }
    }

    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, const unordered_map<size_t, bool>
			      &anchor_candidates, vector<size_t>
			      *anchor_indices, Eigen::MatrixXd *A) {
	ASSERT(rank <= M.rows() && rank <= M.cols(), "Rank > dimension");

	// Identify anchor rows by finding vertices of a convex hull.
	find_vertex_rows(M, rank, anchor_candidates, anchor_indices);

	// Use the identified anchor rows to extract the sub-component matrix.
	extract_matrix(M, rank, max_num_updates, stopping_threshold,
		       verbose, *anchor_indices, A);
    }

    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, const vector<size_t>
			      &anchor_indices, Eigen::MatrixXd *A) {
	ASSERT(rank <= M.rows() && rank <= M.cols(), "Rank > dimension");

	// Use the given anchor rows to extract the sub-component matrix.
	extract_matrix(M, rank, max_num_updates, stopping_threshold,
		       verbose, anchor_indices, A);
    }

    void anchor_factorization(const Eigen::MatrixXd &M, size_t rank,
			      size_t max_num_updates, double stopping_threshold,
			      bool verbose, vector<size_t> *anchor_indices,
			      Eigen::MatrixXd *A) {
	unordered_map<size_t, bool> anchor_candidates;
	for (size_t row = 0; row < M.rows(); ++row) {
	    anchor_candidates[row] = true;  // Consider all rows for anchors.
	}
	anchor_factorization(M, rank, max_num_updates, stopping_threshold,
			     verbose, anchor_candidates, anchor_indices, A);
    }
}  // namespace optimize
