#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define NX 500  // x: columns
#define NY 500  // y: rows
#define NZ 500  // z: planes (depth)

#define CHECK(call)                                                            \
do {                                                                           \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                          \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(1);                                                               \
    }                                                                          \
} while (0)

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

__global__ void mulElem3D(int *a, int *b, int *c, int nx, int ny, int nz) {
    // TODO 1: the x, y and z coordinates of this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // TODO 2: only compute if x, y AND z are all inside the volume
    if (x < nx && y < ny && z < nz) {
        // TODO 3: flat index of (x, y, z): ((z * ny + y) * nx + x)
        size_t idx = ((size_t)z * ny + y) * nx + x;
        c[idx] = a[idx] * b[idx];
    }
}

void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}

int main(void) {
    size_t count = (size_t)NX * NY * NZ;  // 125,000,000 elements
    size_t size = count * sizeof(int);     // 500 MB per array
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    double dStart, dElaps;

    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    a = (int*)malloc(size); random_ints(a, count);
    b = (int*)malloc(size); random_ints(b, count);
    c = (int*)malloc(size);

    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // 8 x 8 x 8 = 512 threads per block
    dim3 threadsPerBlock(8, 8, 8);

    // TODO 4: blocks needed in x, y and z, rounding UP
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    printf("Grid: %d x %d x %d blocks\n", numBlocks.x, numBlocks.y, numBlocks.z);

    dStart = cpuSecond();
    // TODO 5: launch mulElem3D with <<<numBlocks, threadsPerBlock>>>
    mulElem3D<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, NX, NY, NZ);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    // TODO 6: print the last 1000 elements of C in memory order
    // Starts at z = 499, y = 498, x = 0 up to z = 499, y = 499, x = 499
    printf("\nPrinting sample of last 1000 elements:\n");
    size_t start_idx = count - 1000;
    size_t z_start = start_idx / ((size_t)NY * NX);
    size_t rem = start_idx % ((size_t)NY * NX);
    size_t y_start = rem / NX;
    size_t x_start = rem % NX;

    printf("First of last 1000: C[%zu][%zu][%zu] = %d x %d = %d\n",
           z_start, y_start, x_start, a[start_idx], b[start_idx], c[start_idx]);
    printf("Last result: C[499][499][499] = %d x %d = %d\n",
           a[count - 1], b[count - 1], c[count - 1]);

    // TODO 7: check all elements on the CPU
    int mismatches = 0;
    for (size_t i = 0; i < count; i++) {
        if (c[i] != a[i] * b[i]) {
            mismatches++;
        }
    }

    printf("Verification: %d mismatches out of %zu\n", mismatches, count);
    printf("Time taken %.7f\n", dElaps);

    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return 0;
}
