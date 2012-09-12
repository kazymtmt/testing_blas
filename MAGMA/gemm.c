#ifdef TESTING_DOUBLE_PRECISION
typedef double precision;
#define TESTBLAS_GEMM magmablas_dgemm
#define CBLAS_GEMM cblas_dgemm
#else
typedef float precision;
#define TESTBLAS_GEMM magmablas_sgemm
#define CBLAS_GEMM cblas_sgemm
#endif

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cblas.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <magma.h>

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

  const char transa = (TransA == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;
  const char transb = (TransB == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;

  //cublasStatus stat;
  precision *bufA, *bufB, *bufC;

  // Initialize CUBLAS
  if(CUBLAS_STATUS_SUCCESS != cublasInit()) {
    fprintf(stderr, "cublasInit(): failed\n");
    exit(1);
  }

  precision *A, *B, *C, *D;
  magma_malloc_cpu((void**) &A, max_size*max_size*sizeof(precision));
  magma_malloc_cpu((void**) &B, max_size*max_size*sizeof(precision));
  magma_malloc_cpu((void**) &C, max_size*max_size*sizeof(precision));
  magma_malloc_cpu((void**) &D, max_size*max_size*sizeof(precision));
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
    magma_malloc((void**)(&bufA), M*K*sizeof(*A));
    magma_malloc((void**)(&bufB), K*N*sizeof(*B));
    magma_malloc((void**)(&bufC), M*N*sizeof(*C));

    int i;
    for(i = 0; i < M*K; i++) A[i] = (precision)rand()/(RAND_MAX);
    for(i = 0; i < K*N; i++) B[i] = (precision)rand()/(RAND_MAX);
    for(i = 0; i < M*N; i++) C[i] = (precision)rand()/(RAND_MAX);

#ifdef TESTING_DOUBLE_PRECISION
    magma_dsetmatrix(M, K, A, lda, bufA, lda);
    magma_dsetmatrix(K, N, B, ldb, bufB, ldb);
    magma_dsetmatrix(M, N, C, ldc, bufC, ldc);
    putchar('D');
#else
    magma_ssetmatrix(M, K, A, lda, bufA, lda);
    magma_ssetmatrix(K, N, B, ldb, bufB, ldb);
    magma_ssetmatrix(M, N, C, ldc, bufC, ldc);
    putchar('S');
#endif
    printf("%c%c%c : %4d %4d %4d",
        (Order == CblasColMajor) ? 'C' : 'R',
        (TransA == CblasNoTrans) ? 'N' : 'T',
        (TransB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);
    double gflops = 2.0 * (double)M * (double)N * (double)K;
    gflops /= 1e9;
    magma_timestr_t  start, end;
    if (error_check) {
      memcpy(D, C, M*N*sizeof(*C));
      if (Order == CblasColMajor) {
        TESTBLAS_GEMM(transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc);
      } else {
        TESTBLAS_GEMM(transb, transa, N, M, K, alpha, bufB, ldb, bufA, lda, beta, bufC, ldc);
      }
#ifdef TESTING_DOUBLE_PRECISION
      magma_dgetmatrix(M, N, bufC, ldc, C, ldc);
#else
      magma_sgetmatrix(M, N, bufC, ldc, C, ldc);
#endif
      CBLAS_GEMM(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, D, ldc);
      error_check_gemm(C, D, M, N);
    } else {
      start = get_current_time();
      if (Order == CblasColMajor) {
        TESTBLAS_GEMM(transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc);
      } else {
        TESTBLAS_GEMM(transb, transa, N, M, K, alpha, bufB, ldb, bufA, lda, beta, bufC, ldc);
      }
      end = get_current_time();
      double comp_time = GetTimerValue(start, end);
      //printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    }

    start = get_current_time();
    if (Order == CblasColMajor) {
      TESTBLAS_GEMM(transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc);
    } else {
      TESTBLAS_GEMM(transb, transa, N, M, K, alpha, bufB, ldb, bufA, lda, beta, bufC, ldc);
    }
    end = get_current_time();
    double comp_time = GetTimerValue(start, end) / 1e3;
    printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    fflush(stdout);

    magma_free(bufA);
    magma_free(bufB);
    magma_free(bufC);
  }
  cublasShutdown();
  free(A); free(B); free(C);

  return ret;
}
