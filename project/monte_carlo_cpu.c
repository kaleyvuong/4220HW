#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double pi_estimate(long n_samples) {
    long inside = 0;
    for (long i = 0; i < n_samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x*x + y*y <= 1.0) inside++;
    }
    return 4.0 * inside / n_samples;
}

int main() {
    srand(42);
    long n = 100000000L;  // 10^8 samples

    clock_t start = clock();
    double pi = pi_estimate(n);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("π ≈ %.6f  |  samples: %ld  |  time: %.3fs\n", pi, n, elapsed);
    return 0;
}