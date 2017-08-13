#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <time.h>


#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl


int main()
{

    const uint dim_x = 2;
    const uint dim_u = 1;
    Eigen::MatrixXd A(dim_x, dim_x);
    Eigen::MatrixXd B(dim_x, dim_u);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(dim_x, dim_x);
    Eigen::MatrixXd R(dim_u, dim_u);
    A <<
        0.1, 1.0,
        1.0, 2.0;
    B <<
        0.0,
        1.0;

    R(0,0) = 1.0;

    Eigen::MatrixXd AT = A.transpose();
    Eigen::MatrixXd BT = B.transpose();


    /* iteration based Riccati solution */
    double tolerance = 1E-5;
    uint iter_max = 1000;
    double dt = 0.1;
    std::vector<double> diff_mat;
    Eigen::MatrixXd P = Q;

    clock_t start = clock();
    for(uint i = 0; i < iter_max; ++i){
        Eigen::MatrixXd P_next = P +
            (P * A + AT * P - P * B  * R.inverse() * BT * P + Q) * dt;
        double diff = fabs((P_next - P).maxCoeff());
        diff_mat.push_back(diff);
        // PRINT_MAT(P_next - P);
        // std::cout << "diff = " << diff << std::endl;

        P = P_next;

        if(diff < tolerance){
            std::cout << i << std::endl;
            break;
        }

    }
    clock_t end = clock();
    std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    PRINT_MAT(P);



    /* eigen decomposition method (Arimoto-Potter algorithm) */
    start = clock();

    // Hamilton matrix
    Eigen::MatrixXd Ham(2 * dim_x, 2 * dim_x);
    Ham <<
        A, -B * R.inverse() * BT,
        -Q, -AT;

    Eigen::EigenSolver<Eigen::MatrixXd> Eigs(Ham);
/*
    std::cout << "eigen values：\n"
              << Eigs.eigenvalues() << std::endl;
    std::cout << "eigen vectors：\n"
              << Eigs.eigenvectors() << std::endl;
*/

    // Extract only stable eigenvectors into eigvec
    std::vector<int> i_vec;
    Eigen::MatrixXd eigvec(2*dim_x, dim_x);
    for(int i = 0; i < 2 * dim_x; ++i){
        if(Eigs.eigenvalues()[i].real() < 0){
            static int j = 0;
            eigvec.col(j) = Eigs.eigenvectors().block(0,i,2*dim_x,1).real();
            ++j;
        }
    }

    Eigen::MatrixXd Vs_1(dim_x, dim_x);
    Eigen::MatrixXd Vs_2(dim_x, dim_x);
    Vs_1 = eigvec.block(0,0,dim_x,dim_x).real();
    Vs_2 = eigvec.block(dim_x,0,dim_x,dim_x).real();
    P = Vs_2 * Vs_1.inverse();

    end = clock();
    std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    PRINT_MAT(P);

    return 0;

}
