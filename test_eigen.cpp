#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <time.h>


#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl


int main()
{


    const uint dim_x = 4;
    const uint dim_u = 1;
    Eigen::Matrix<double, dim_x, dim_x> A;
    Eigen::Matrix<double, dim_x, dim_u> B;
    Eigen::Matrix<double, dim_x, dim_x> Q;
    Eigen::Matrix<double, dim_u, dim_u> R;
    Eigen::Matrix<double, dim_u, dim_x> K;
    /*
    A <<
        0.1, 1.0,
        1.0, 2.0;
    B <<
        0.0,
        1.0;
    */
    A <<
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0;

    B <<
        0.0,
        0.0,
        0.0,
        0.0;

    A(1,1) = - 83.0;
    A(1,2) = 16.0;
    A(3,3) = - 83.0;
    B(1,0) = 20.0;
    B(3,0) = 2.6;

    Q <<
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0;



    R(0,0) = 1.0;

    PRINT_MAT(A);
    PRINT_MAT(B);
    PRINT_MAT(Q);
    PRINT_MAT(R);



    Eigen::Matrix<double, dim_x, dim_x> AT = A.transpose();
    Eigen::Matrix<double, dim_u, dim_x> BT = B.transpose();


    /* == iteration based Riccati solution == */
    std::cout << "-- Iteration based method --" << std::endl;

    double tolerance = 1E-5;
    uint iter_max = 100000;
    double dt = 0.01;

    // discritization
    Eigen::Matrix<double, dim_x, dim_x> I = Eigen::MatrixXd::Identity(dim_x, dim_x);
    Eigen::Matrix<double, dim_x, dim_x> Ad;
    Ad = (I + 0.5 * dt * A) * (I - 0.5 * dt * A).inverse();
    Eigen::Matrix<double, dim_x, dim_u> Bd;
    Bd = B * dt;
    Eigen::Matrix<double, dim_x, dim_x> AdT = Ad.transpose();
    Eigen::Matrix<double, dim_u, dim_x> BdT = Bd.transpose();


    std::vector<double> diff_mat;
    Eigen::Matrix<double, dim_x, dim_x> P = Q;

    clock_t start = clock();
    for(uint i = 0; i < iter_max; ++i){
        // -- continuous solver --
        /*
          Eigen::Matrix<double, dim_x, dim_x> P_next = P +
            (P * A + AT * P - P * B  * R.inverse() * BT * P + Q) * dt;
        */
        // -- discritized solver --
        Eigen::Matrix<double, dim_x, dim_x> P_next =
            AdT * P * Ad - AdT * P * Bd * (R + BdT * P * Bd).inverse() * BdT * P * Ad + Q;

        double diff = fabs((P_next - P).maxCoeff());
        diff_mat.push_back(diff);
        P = P_next;
        if(diff < tolerance){
            std::cout << "iteration mumber = " << i << std::endl;
            break;
        }

    }
    clock_t end = clock();
    std::cout << "computation time = " <<
        (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    PRINT_MAT(P);
    K = (R + BdT * P * Bd).inverse() * BdT * P * Ad;
    PRINT_MAT(K);




    /* == eigen decomposition method (Arimoto-Potter algorithm) == */
    std::cout << "-- Eigen decomposition mathod --" << std::endl;
    start = clock();

    // Hamilton matrix
    Eigen::Matrix<double, 2*dim_x, 2*dim_x> Ham;
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
    Eigen::Matrix<std::complex<double>, 2*dim_x, dim_x> eigvec;
    int j = 0;

    for(int i = 0; i < 2 * dim_x; ++i){
        if(Eigs.eigenvalues()[i].real() < 0){
            eigvec.col(j) = Eigs.eigenvectors().block(0, i, 2*dim_x, 1);
            ++j;
        }
    }

    Eigen::Matrix<std::complex<double>, dim_x, dim_x> Vs_1;
    Eigen::Matrix<std::complex<double>, dim_x, dim_x> Vs_2;
    Vs_1 = eigvec.block(0,0,dim_x,dim_x);
    Vs_2 = eigvec.block(dim_x,0,dim_x,dim_x);
    P = (Vs_2 * Vs_1.inverse()).real();

    end = clock();
    std::cout << "computation time = " <<
        (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
    PRINT_MAT(P);
    // Eigen::Matrix<double, dim_u, dim_x> K;
    K = R.inverse() * BT * P;

    PRINT_MAT(K);

    return 0;

}
