#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define WIDTH 800   // columns
#define HEIGHT 600  // rows

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

__global__ void PictureKernel(unsigned char *d_Pin, unsigned char *d_Pout, int height, int width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < height) && (Col < width)) {
        int v = 2 * d_Pin[Row * width + Col];
        d_Pout[Row * width + Col] = (v > 255) ? 255 : v;
    }
}

int main(void) {
    size_t count = (size_t)WIDTH * HEIGHT;
    size_t size = count * sizeof(unsigned char);
    unsigned char *h_in, *h_out;
    unsigned char *d_in, *d_out;
    double dStart, dElaps;

    h_in = (unsigned char *)malloc(size);
    h_out = (unsigned char *)malloc(size);

    CHECK(cudaMalloc((void **)&d_in, size));
    CHECK(cudaMalloc((void **)&d_out, size));

    // TODO 1: give every pixel of h_in a random value between 0 and 255
    for (size_t i = 0; i < count; i++) {
        h_in[i] = (unsigned char)(rand() % 256);
    }

    CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // TODO 2: 16 threads per block in BOTH x and y
    dim3 threadsPerBlock(16, 16);

    // TODO 3: blocks needed in x (columns) and in y (rows), rounding UP
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Grid: %d x %d blocks of %d x %d threads\n",
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    dStart = cpuSecond();
    // TODO 4: launch PictureKernel with <<<numBlocks, threadsPerBlock>>>
    PictureKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, HEIGHT, WIDTH);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    dElaps = cpuSecond() - dStart;

    CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // TODO 5: print top-left 50x50 portion: original then updated
    printf("\n--- Top-left 50x50 Original ---\n");
    for (int r = 0; r < 50; r++) {
        for (int c = 0; c < 50; c++) {
            printf("%3d ", h_in[r * WIDTH + c]);
        }
        printf("\n");
    }

    printf("\n--- Top-left 50x50 Updated ---\n");
    for (int r = 0; r < 50; r++) {
        for (int c = 0; c < 50; c++) {
            printf("%3d ", h_out[r * WIDTH + c]);
        }
        printf("\n");
    }

    // TODO 6: check every pixel on the CPU (capped at 255)
    int mismatches = 0;
    int capped = 0;
    for (size_t i = 0; i < count; i++) {
        int expected = 2 * h_in[i];
        if (expected > 255) {
            expected = 255;
            capped++;
        }
        if (h_out[i] != (unsigned char)expected) {
            mismatches++;
        }
    }

    printf("\nVerification: %d mismatches out of %zu\n", mismatches, count);
    printf("Pixels capped at 255: %d\n", capped);
    printf("Time taken %.7f\n", dElaps);

    free(h_in); free(h_out);
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out));
    return 0;
}
