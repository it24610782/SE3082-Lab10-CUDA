#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

#define CHECK(call)                                                            \
do {                                                                           \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                          \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(1);                                                               \
    }                                                                          \
} while (0)

__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);
    double dStart, dElaps;

    // TODO 1: wrap each of the three cudaMalloc calls in CHECK(...)
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    a = (int*)malloc(size); random_ints(a, N);
    b = (int*)malloc(size); random_ints(b, N);
    c = (int*)malloc(size);

    // TODO 2: wrap both host-to-device cudaMemcpy calls in CHECK(...)
    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    dStart = cpuSecond();
    add<<<N, 1>>>(d_a, d_b, d_c);

    // TODO 3: add CHECK(cudaGetLastError()); here, and wrap cudaDeviceSynchronize()
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dElaps = cpuSecond() - dStart;

    // TODO 4: wrap the device-to-host cudaMemcpy in CHECK(...)
    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    for (int r = 0; r < 5; r++)
        printf("%d + %d = %d\n", a[r], b[r], c[r]);

    int errors = 0;
    for (int r = 0; r < N; r++)
        if (c[r] != a[r] + b[r]) errors++;
    printf("Verification: %d mismatches out of %d\n", errors, N);
    printf("Time taken %.7f\n", dElaps);

    free(a); free(b); free(c);

    // TODO 5: wrap the three cudaFree calls in CHECK(...)
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}
