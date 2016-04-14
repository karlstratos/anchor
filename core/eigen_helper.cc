// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "eigen_helper.h"

#include <random>

namespace eigen_helper {
    void compute_pseudoinverse(const Eigen::MatrixXd &matrix,
			       Eigen::MatrixXd *matrix_pseudoinverse) {
	double tol = 1e-6;
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU |
					      Eigen::ComputeThinV);
	Eigen::VectorXd inverse_singular_values(svd.singularValues().size());
	for (size_t i = 0; i < svd.singularValues().size(); ++i) {
	    if (svd.singularValues()(i) > tol) {
		inverse_singular_values(i) = 1.0 / svd.singularValues()(i);
	    } else {
		inverse_singular_values(i) = 0.0;
	    }
	}
	(*matrix_pseudoinverse) = svd.matrixV() *
	    inverse_singular_values.asDiagonal() * svd.matrixU().transpose();
    }

    void extend_orthonormal_basis(const Eigen::VectorXd &v,
				  Eigen::MatrixXd *orthonormal_basis) {
	Eigen::MatrixXd orthogonal_projection =
	    (*orthonormal_basis) * (*orthonormal_basis).transpose();
	Eigen::VectorXd projected_v = orthogonal_projection * v;
	Eigen::VectorXd new_direction = v - projected_v;
	new_direction /= new_direction.norm();

	orthonormal_basis->conservativeResize(orthonormal_basis->rows(),
					     orthonormal_basis->cols() + 1);
	(*orthonormal_basis).col(orthonormal_basis->cols() - 1) = new_direction;
    }

    void find_range(const Eigen::MatrixXd &matrix,
		    Eigen::MatrixXd *orthonormal_basis) {
	ASSERT(matrix.cols() > 0, "Matrix has 0 columns");

	// Find the first basis element.
	(*orthonormal_basis) = matrix.col(0);
	(*orthonormal_basis) /= orthonormal_basis->norm();

	// Find the remaining basis elements.
	for (size_t col = 1; col < matrix.cols(); ++col) {
	    if (col >= matrix.rows()) {
		// The dimension of the range is at most the number of rows.
		break;
	    }
	    extend_orthonormal_basis(matrix.col(col), orthonormal_basis);
	}
    }

    void compute_pca(const Eigen::MatrixXd &original_sample_rows,
		     Eigen::MatrixXd *rotated_sample_rows,
		     Eigen::MatrixXd *rotation_matrix,
		     Eigen::VectorXd *variances) {
	// Center each dimension (column).
	Eigen::MatrixXd centered = original_sample_rows;
	Eigen::VectorXd averages = centered.colwise().sum() / centered.rows();
	for (size_t i = 0; i < centered.cols(); ++i) {
	    Eigen::VectorXd average_column =
		Eigen::VectorXd::Constant(centered.rows(), averages(i));
	    centered.col(i) -= average_column;
	}

	// Perform SVD.
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU |
					      Eigen::ComputeThinV);

	// Set values.
	*rotated_sample_rows =
	    svd.matrixU() * svd.singularValues().asDiagonal();
	*rotation_matrix = svd.matrixV();
	(*variances).resize(svd.singularValues().size());
	for (size_t i = 0; i < svd.singularValues().size(); ++i) {
	    (*variances)(i) =
		pow(svd.singularValues()(i), 2) / (centered.rows() - 1);
	}
    }

    void generate_random_projection(size_t original_dimension,
				    size_t reduced_dimension,
				    Eigen::MatrixXd *projection_matrix) {
	projection_matrix->resize(original_dimension, reduced_dimension);
	random_device device;
	default_random_engine engine(device());
	// Indyk and Motwani (1998)
	normal_distribution<double> normal(0.0, 1.0 / reduced_dimension);
	for (size_t row = 0; row < original_dimension; ++row) {
	    for (size_t col = 0; col < reduced_dimension; ++col) {
		(*projection_matrix)(row, col) = normal(engine);
	    }
	}
    }
}  // namespace eigen_helper
