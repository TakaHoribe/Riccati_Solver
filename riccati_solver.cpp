/*
// solvers for Algebraic Riccati equation
// - Iteration (continuous)
// - Iteration (discrete)
// - Arimoto-Potter
//
// author: Horibe Takamasa
*/

#include <Eigen/Dense>
#include <iostream>
#include <time.h>
#include <vector>

#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl

bool solveRiccatiIterationC(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                            const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R,
                            Eigen::MatrixXd &P, const double dt = 0.001,
                            const double &tolerance = 1.E-5,
                            const uint iter_max = 100000) {
  P = Q; // initialize

  Eigen::MatrixXd P_next;

  Eigen::MatrixXd AT = A.transpose();
  Eigen::MatrixXd BT = B.transpose();
  Eigen::MatrixXd Rinv = R.inverse();

  double diff;
  for (uint i = 0; i < iter_max; ++i) {
    P_next = P + (P * A + AT * P - P * B * Rinv * BT * P + Q) * dt;
    diff = fabs((P_next - P).maxCoeff());
    P = P_next;
    if (diff < tolerance) {
      std::cout << "iteration mumber = " << i << std::endl;
      return true;
    }
  }
  return false; // over iteration limit
}

bool solveRiccatiIterationD(const Eigen::MatrixXd &Ad,
                            const Eigen::MatrixXd &Bd, const Eigen::MatrixXd &Q,
                            const Eigen::MatrixXd &R, Eigen::MatrixXd &P,
                            const double &tolerance = 1.E-5,
                            const uint iter_max = 100000) {
  P = Q; // initialize

  Eigen::MatrixXd P_next;

  Eigen::MatrixXd AdT = Ad.transpose();
  Eigen::MatrixXd BdT = Bd.transpose();
  Eigen::MatrixXd Rinv = R.inverse();

  double diff;
  for (uint i = 0; i < iter_max; ++i) {
    // -- discrete solver --
    P_next = AdT * P * Ad -
             AdT * P * Bd * (R + BdT * P * Bd).inverse() * BdT * P * Ad + Q;

    diff = fabs((P_next - P).maxCoeff());
    P = P_next;
    if (diff < tolerance) {
      std::cout << "iteration mumber = " << i << std::endl;
      return true;
    }
  }
  return false; // over iteration limit
}

bool solveRiccatiArimotoPotter(const Eigen::MatrixXd &A,
                               const Eigen::MatrixXd &B,
                               const Eigen::MatrixXd &Q,
                               const Eigen::MatrixXd &R, Eigen::MatrixXd &P) {

  const uint dim_x = A.rows();
  const uint dim_u = B.cols();

  // set Hamilton matrix
  Eigen::MatrixXd Ham = Eigen::MatrixXd::Zero(2 * dim_x, 2 * dim_x);
  Ham << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();

  // calc eigenvalues and eigenvectors
  Eigen::EigenSolver<Eigen::MatrixXd> Eigs(Ham);

  // check eigen values
  // std::cout << "eigen values：\n" << Eigs.eigenvalues() << std::endl;
  // std::cout << "eigen vectors：\n" << Eigs.eigenvectors() << std::endl;

  // extract stable eigenvectors into 'eigvec'
  Eigen::MatrixXcd eigvec = Eigen::MatrixXcd::Zero(2 * dim_x, dim_x);
  for (int i = 0; i < 2 * dim_x; ++i) {
    if (Eigs.eigenvalues()[i].real() < 0.) {
      static int j = 0;
      eigvec.col(j) = Eigs.eigenvectors().block(0, i, 2 * dim_x, 1);
      ++j;
    }
  }

  // calc P with stable eigen vector matrix
  Eigen::MatrixXcd Vs_1, Vs_2;
  Vs_1 = eigvec.block(0, 0, dim_x, dim_x);
  Vs_2 = eigvec.block(dim_x, 0, dim_x, dim_x);
  P = (Vs_2 * Vs_1.inverse()).real();

  return true;
}

int main() {

  const uint dim_x = 4;
  const uint dim_u = 1;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dim_x, dim_x);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(dim_x, dim_u);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(dim_x, dim_x);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(dim_u, dim_u);
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dim_x, dim_x);

  A(0, 1) = 1.0;
  A(1, 1) = -15.0;
  A(1, 2) = 10.0;
  A(2, 3) = 1.0;
  A(3, 3) = -15.0;
  B(1, 0) = 10.0;
  B(3, 0) = 1.0;

  Q(0, 0) = 1.0;
  Q(2, 2) = 1.0;
  Q(3, 3) = 2.0;

  R(0, 0) = 1.0;

  PRINT_MAT(A);
  PRINT_MAT(B);
  PRINT_MAT(Q);
  PRINT_MAT(R);

  /* == iteration based Riccati solution (continuous) == */
  std::cout << "-- Iteration based method (continuous) --" << std::endl;
  clock_t start = clock();
  solveRiccatiIterationC(A, B, Q, R, P);
  clock_t end = clock();
  std::cout << "computation time = " << (double)(end - start) / CLOCKS_PER_SEC
            << "sec." << std::endl;
  PRINT_MAT(P);

  /* == iteration based Riccati solution (discrete) == */
  // discretization
  const double dt = 0.001;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_x, dim_x);
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(dim_x, dim_x);
  Ad = (I + 0.5 * dt * A) * (I - 0.5 * dt * A).inverse();
  Eigen::MatrixXd Bd;
  Bd = B * dt;

  std::cout << "-- Iteration based method (discrete)--" << std::endl;
  start = clock();
  solveRiccatiIterationD(Ad, Bd, Q, R, P);
  end = clock();
  std::cout << "computation time = " << (double)(end - start) / CLOCKS_PER_SEC
            << "sec." << std::endl;
  PRINT_MAT(P);

  /* == eigen decomposition method (Arimoto-Potter algorithm) == */
  std::cout << "-- Eigen decomposition mathod --" << std::endl;
  start = clock();
  solveRiccatiArimotoPotter(A, B, Q, R, P);
  end = clock();
  std::cout << "computation time = " << (double)(end - start) / CLOCKS_PER_SEC
            << "sec." << std::endl;
  PRINT_MAT(P);

  return 0;
}
