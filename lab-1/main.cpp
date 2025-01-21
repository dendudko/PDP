#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include <fstream>
#include <vector>
#include <functional>

const double N = 100'000'000'0;
const size_t experiments = 10;

double f(double x) {
    return x * x - 1;
}

double integrateSequential(double a, double b, size_t numThreads) {
    double sum = 0;
    double dx = (b - a) / N;

    for (size_t i = 0; i < N; ++i) {
        sum += f(a + i * dx);
    }

    return dx * sum;
}

double integrateParallel(double a, double b, size_t numThreads) {
    double sum = 0;
    double dx = (b - a) / N;

#pragma omp parallel num_threads(numThreads)
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        double threadSum = 0;

        for (size_t i = t; i < N; i += T) {
            threadSum += f(a + i * dx);
        }

#pragma omp critical
        {
            sum += threadSum;
        }
    }

    return dx * sum;
}

double runExperiment(std::function<double(double, double, size_t)> integrationFunc, double a, double b, size_t numThreads) {
    double totalTime = 0;
    double result = 0;

    for (size_t i = 0; i < experiments; ++i) {
        double t1 = omp_get_wtime();
        result = integrationFunc(a, b, numThreads);
        double t2 = omp_get_wtime() - t1;

        totalTime += t2;
    }

    double avgTime = totalTime / experiments;
    std::cout << "Threads: " << numThreads << ", Average Time: " << avgTime << "s, Result: " << result  << std::endl;
    return avgTime;
}

int main() {
    std::ofstream output("./output.csv");

    if (!output.is_open()) {
        std::cout << "Couldn't open file!\n";
        return -1;
    }

    const size_t threadCount = std::thread::hardware_concurrency();
    const double a = 0;
    const double b = 1;

    std::vector<double> times(threadCount + 1);
    std::vector<double> values(threadCount + 1);

    std::cout << "Sequential\n";
    times[0] = runExperiment(integrateSequential, a, b, 0);

    std::cout << "Parallel\n";
    for (size_t i = 1; i <= threadCount; i++) {
        times[i] = runExperiment(integrateParallel, a, b, i);
    }

    output << "Threads,Average Time (s)\n";
    for (size_t i = 0; i <= threadCount; i++) {
        output << i << "," << times[i] << "\n";
    }

    output.close();
    return 0;
}
