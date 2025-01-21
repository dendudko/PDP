#include <assert.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>

using namespace std;

const size_t matrixSize = 64 * (1 << 4);
const int experiments = 10;

void mulMatrix(
    double *A,
    size_t cA,
    size_t rA,
    const double *B,
    size_t cB,
    size_t rB,
    const double *C,
    size_t cC,
    size_t rC
) {
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    for (size_t i = 0; i < cA; i++) {
        for (size_t j = 0; j < rA; j++) {
            A[i * rA + j] = 0;
            for (size_t k = 0; k < cB; k++) {
                A[i * rA + j] += B[k * rB + j] * C[i * rC + k];
            }
        }
    }
}

void mulMatrix256(
    double *A,
    const double *B,
    const double *C,
    size_t cA,
    size_t rA,
    size_t cB,
    size_t rB,
    size_t cC,
    size_t rC
) {
    assert(cB == rC && cA == cC && rA == rB);
    assert((cA & 0x3f) == 0);

    const size_t values_per_operation = 4;

    for (size_t i = 0; i < rB / values_per_operation; i++) {
        for (size_t j = 0; j < cC; j++) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < rC; k++) {
                __m256d bCol = _mm256_loadu_pd(B + k * cB + i * values_per_operation);
                __m256d broadcasted = _mm256_set1_pd(C[j * rC + k]);
                // sum = _mm256_fmadd_pd(bCol, broadcasted, sum);
                __m256d mulResult = _mm256_mul_pd(bCol, broadcasted);
                sum = _mm256_add_pd(sum, mulResult);
            }

            _mm256_storeu_pd(A + j * rA + i * values_per_operation, sum);
        }
    }
}

vector<double> getPermutationMatrix(size_t n) {
    vector<double> matrix(n * n);
    vector<size_t> permut(n);

    for (size_t i = 0; i < n; i++) {
        permut[i] = i;
        swap(permut[i], permut[rand() % (i + 1)]);
    }

    for (size_t c = 0; c < n; c++) {
        matrix[c * n + permut[c]] = 1;
    }

    return matrix;
}

vector<double> getIdentityMatrix(size_t n) {
    vector<double> matrix(n * n);

    for (size_t c = 0; c < n; c++) {
        matrix[c * n + c] = 1;
    }

    return matrix;
}

int main(int argc, char **argv) {
    srand(time(nullptr));

    std::ofstream output("../output.csv");

    if (!output.is_open()) {
        std::cout << "Couldn't open file!\n";
        return -1;
    }

    vector<double> A(matrixSize * matrixSize), D(matrixSize * matrixSize);

    auto identity = getIdentityMatrix(matrixSize);
    auto permutation = getPermutationMatrix(matrixSize);

    vector<double> B = identity;
    vector<double> C = permutation;

    vector<double> scalar_times(experiments);
    vector<double> vector_times(experiments);

    for (int experiment = 0; experiment < experiments; ++experiment) {
        auto t1 = chrono::steady_clock::now();
        mulMatrix(A.data(), matrixSize, matrixSize,
                  B.data(), matrixSize, matrixSize,
                  C.data(), matrixSize, matrixSize);
        auto t2 = chrono::steady_clock::now();
        scalar_times[experiment] = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

        t1 = chrono::steady_clock::now();
        mulMatrix256(D.data(), B.data(), C.data(), matrixSize, matrixSize, matrixSize, matrixSize, matrixSize,
                     matrixSize);
        t2 = chrono::steady_clock::now();
        vector_times[experiment] = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

        if (!std::memcmp(static_cast<void *>(A.data()),
                         static_cast<void *>(D.data()),
                         matrixSize * matrixSize * sizeof(double))) {
            cout << "Experiment " << experiment << ": The results of matrix multiplication are the same! \n" <<
                    "Scalar time: " << scalar_times[experiment] << " ms\n" << "Vector time: " << vector_times[
                        experiment] << " ms\n\n";
        }
    }

    double avg_scalar_time = 0;
    double avg_vector_time = 0;

    for (int test = 0; test < experiments; ++test) {
        avg_scalar_time += scalar_times[test];
        avg_vector_time += vector_times[test];
    }

    avg_scalar_time /= experiments;
    avg_vector_time /= experiments;

    output << "Experiment,Scalar,Vector\n";
    for (int test = 0; test < experiments; ++test) {
        output << test << "," << scalar_times[test] << "," << vector_times[test] << std::endl;
    }
    output << experiments << "," << avg_scalar_time << "," << avg_vector_time;

    cout << "Average Scalar Time: " << avg_scalar_time << " ms\n";
    cout << "Average Vector Time: " << avg_vector_time << " ms\n";

    output.close();
}
