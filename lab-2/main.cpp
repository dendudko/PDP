#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <fstream>
using namespace std::chrono;

const int cols = 1 << 15;
const int rows = 1 << 15;
const size_t batch = 4;
const size_t experiments = 10;

void addMatrix(double *A, const double *B, const double *C, size_t colsc, size_t rowsc) {
    for (size_t i = 0; i < colsc * rowsc; i++) {
        A[i] = B[i] + C[i];
    }
}

void addMatrix256(double *A, const double *B, const double *C, size_t colsc, size_t rowsc) {
    for (size_t i = 0; i < rowsc * colsc / batch; i++) {
        __m256d b = _mm256_loadu_pd(&(B[i * batch]));
        __m256d c = _mm256_loadu_pd(&(C[i * batch]));
        __m256d a = _mm256_add_pd(b, c);

        _mm256_storeu_pd(&(A[i * batch]), a);
    }
}

std::vector<double> B(cols * rows, 1), C(cols * rows, -2), A(cols * rows);

int main(int argc, char **argv) {
    std::ofstream output("./output.csv");

    if (!output.is_open()) {
        std::cout << "Couldn't open file!\n";
        return -1;
    }

    std::vector<double> s_times(experiments);

    for (int experiment = 0; experiment < experiments; ++experiment) {
        auto t1 = steady_clock::now();
        addMatrix(A.data(), B.data(), C.data(), cols, rows);
        auto t2 = steady_clock::now();
        s_times[experiment] = duration_cast<milliseconds>(t2 - t1).count();

        std::fill_n(A.data(), rows * cols, 0);
        std::fill_n(B.data(), rows * cols, -2);
        std::fill_n(C.data(), rows * cols, 1);
    }

    double totalScalarTime = 0;
    for (double time: s_times) {
        totalScalarTime += time;
    }
    double avgScalarTime = totalScalarTime / s_times.size();

    std::cout << "Average Scalar Time: " << avgScalarTime << " ms\n";

    std::fill_n(A.data(), rows * cols, 0);
    std::fill_n(B.data(), rows * cols, -2);
    std::fill_n(C.data(), rows * cols, 1);

    std::vector<double> v_times(experiments);

    for (int experiment = 0; experiment < experiments; ++experiment) {
        auto t1 = steady_clock::now();
        addMatrix256(A.data(), B.data(), C.data(), cols, rows);
        auto t2 = steady_clock::now();
        v_times[experiment] = duration_cast<milliseconds>(t2 - t1).count();

        std::fill_n(A.data(), rows * cols, 0);
        std::fill_n(B.data(), rows * cols, -2);
        std::fill_n(C.data(), rows * cols, 1);
    }

    double totalVectorTime = 0;
    for (double time: v_times) {
        totalVectorTime += time;
    }
    double avgVectorTime = totalVectorTime / v_times.size();

    std::cout << "Average Vector Time: " << avgVectorTime << " ms\n";

    output << "Experiment,Scalar,Vector\n";
    for (size_t i = 0; i < experiments; i++) {
        output << i << "," << s_times[i] << "," << v_times[i] << "\n";
    }
    output << experiments << "," << avgScalarTime << "," << avgVectorTime << std::endl;

    output.close();
}
