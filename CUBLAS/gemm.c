#ifdef TESTING_DOUBLE_PRECISION
typedef double precision;
#define TESTBLAS_GEMM cublasDgemm_v2
#define CBLAS_GEMM cblas_dgemm
#else
typedef float precision;
#define TESTBLAS_GEMM cublasSgemm_v2
#define CBLAS_GEMM cblas_sgemm
#endif

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cblas.h>
#include <cublas_v2.h>

#include "testing.h"

int main(int argc, char *argv[])
{
  static const precision alpha = 1.3;
  static const precision beta  = 1.7;

  int ret = 0;

  const enum CBLAS_ORDER Order = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const enum CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans : ((atoi(argv[2])==0) ? CblasNoTrans : CblasTrans);
  const enum CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans : ((atoi(argv[3])==0) ? CblasNoTrans : CblasTrans);
  const int max_size = (argc <= 4) ? 32 : atoi(argv[4]);
  const int stride = (argc <= 5) ? 1 : atoi(argv[5]);
  const int error_check = (argc <= 6) ? 0 : atoi(argv[6]);

  const cublasOperation_t transa = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t transb = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  //cublasStatus stat;
  precision *bufA, *bufB, *bufC;

  // Initialize CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  precision *A = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *B = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *C = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *D = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  srand(time(NULL));

  int M;
  for (M = stride; M <= max_size; M += stride) {
    int N, K, lda, ldb, ldc;
    N = K = M;
    //K = 128;
    if (Order == CblasColMajor) {
      lda = (TransA == CblasTrans) ? K : M;
      ldb = (TransB == CblasTrans) ? N : K;
      ldc = M;
    } else {
      lda = (TransA == CblasTrans) ? M : K;;
      ldb = (TransB == CblasTrans) ? K : N;
      ldc = N;
    }

    // Prepare memory objects and place matrices inside them. */
    cudaMalloc((void**)(&bufA), M*K*sizeof(*A));
    cudaMalloc((void**)(&bufB), K*N*sizeof(*B));
    cudaMalloc((void**)(&bufC), M*N*sizeof(*C));

    int i;
    for(i = 0; i < M*K; i++) A[i] = (precision)rand()/(RAND_MAX);
    for(i = 0; i < K*N; i++) B[i] = (precision)rand()/(RAND_MAX);
    for(i = 0; i < M*N; i++) C[i] = (precision)rand()/(RAND_MAX);

    cudaMemcpy(bufA, A, M*K*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(bufB, B, K*N*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemcpy(bufC, C, M*N*sizeof(*C), cudaMemcpyHostToDevice);

#ifdef TESTING_DOUBLE_PRECISION
    putchar('D');
#else
    putchar('S');
#endif
    printf("%c%c%c : %4d %4d %4d",
        (Order == CblasColMajor) ? 'C' : 'R',
        (TransA == CblasNoTrans) ? 'N' : 'T',
        (TransB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);
    double gflops = 2.0 * (double)M * (double)N * (double)K;
    gflops /= 1e9;
    float comp_time;
    if (error_check) {
      memcpy(D, C, M*N*sizeof(*C));
      cudaEventRecord(start, stream);
      if (Order == CblasColMajor) {
        TESTBLAS_GEMM(handle, transa, transb, M, N, K, &alpha, bufA, lda, bufB, ldb, &beta, bufC, ldc);
      } else {
        TESTBLAS_GEMM(handle, transb, transa, N, M, K, &alpha, bufB, ldb, bufA, lda, &beta, bufC, ldc);
      }
      cudaEventRecord(end, stream);
      cudaEventSynchronize(end);
      cudaMemcpy(C, bufC, M*N*sizeof(*C), cudaMemcpyDeviceToHost);
      CBLAS_GEMM(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, D, ldc);
      error_check_gemm(C, D, M, N);
    } else {
      ////comp_time = my_get_current_time();
      //cudaEventRecord(start, stream);
      //if (Order == CblasColMajor) {
      //    TESTBLAS_GEMM(handle, transA, transB, M, N, K, &alpha, bufA, lda, bufB, ldb, &beta, bufC, ldc);
      //} else {
      //    TESTBLAS_GEMM(handle, transB, transA, N, M, K, &alpha, bufB, ldb, bufA, lda, &beta, bufC, ldc);
      //}
      //cudaEventRecord(end, stream);
      //cudaEventSynchronize(end);
      //cudaEventElapsedTime(&comp_time, start, end);
      //comp_time /= 1e3;
      ////comp_time = my_get_current_time() - comp_time;
      //printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    }

    //comp_time = my_get_current_time();
    cudaEventRecord(start, stream);
    if (Order == CblasColMajor) {
      TESTBLAS_GEMM(handle, transa, transb, M, N, K, &alpha, bufA, lda, bufB, ldb, &beta, bufC, ldc);
    } else {
      TESTBLAS_GEMM(handle, transb, transa, N, M, K, &alpha, bufB, ldb, bufA, lda, &beta, bufC, ldc);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&comp_time, start, end);
    comp_time /= 1e3;
    //comp_time = my_get_current_time() - comp_time;
    printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    fflush(stdout);

    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(bufC);
  }
  cudaEventDestroy(end);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  free(A); free(B); free(C);

  return ret;
}
