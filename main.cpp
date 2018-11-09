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
#include "riccati_solver.h"

#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl

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
