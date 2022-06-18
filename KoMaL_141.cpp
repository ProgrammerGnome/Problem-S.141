#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <chrono>

double normal(uint64_t M, uint64_t N, uint64_t P) {
    double Nd = (double)N;
    double sum = 0;

    for (uint64_t i = 0; i < M; ++i)
    {
        sum = sum + Nd / (P + i);
    }

    return sum;
}

// https://tildesites.bowdoin.edu/~ltoma/teaching/cs3225-GIS/fall17/Lectures/openmp.html
double parallel(uint64_t M, uint64_t N, uint64_t P) {
    double Nd = (double)N;
    double local_sum, sum = 0;

#pragma omp parallel private(local_sum) shared(sum) 
    {
        local_sum = 0;

        //the array is distributde statically between threads
#pragma omp for schedule(static,1) 
        for (int64_t i = 0; i < M; ++i) {
            local_sum += Nd / (P + i);
        }

        //each thread calculated its local_sum. ALl threads have to add to
        //the global sum. It is critical that this operation is atomic.
#pragma omp critical 
        sum += local_sum;
    }

    return sum;
}

// https://chryswoods.com/vector_c++/features.html
double simd(uint64_t M, uint64_t N, uint64_t P) {
    double Nd = (double)N;
    double sum = 0;

#pragma omp simd reduction(+:sum)
    for (int64_t i = 0; i < M; ++i)
    {
        sum += Nd / (P + i);
    }

    return sum;
}

int main()
{
    uint64_t M, N, P;

    printf("N: "); scanf("%lld", &N);
    printf("M: "); scanf("%lld", &M);
    printf("P: "); scanf("%lld", &P);

    printf("\t\tSum\t\tTime\n");

    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum = normal(M, N, P);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        printf("Normal\t\t%f\t%10.6f\n", sum, elapsed.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum = parallel(M, N, P);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        printf("OMP parallel\t%f\t%10.6f\n", sum, elapsed.count());
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum = simd(M, N, P);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        printf("OMP SIMD\t%f\t%10.6f\n", sum, elapsed.count());
    }

    return 0;
}