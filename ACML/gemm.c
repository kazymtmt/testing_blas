#ifdef TESTING_DOUBLE_PRECISION
typedef double precision;
#define TESTBLAS_GEMM dgemm
#define CBLAS_GEMM cblas_dgemm
#else
typedef float precision;
#define TESTBLAS_GEMM sgemm
#define CBLAS_GEMM cblas_sgemm
#endif

#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <acml.h>
#include <cblas.h>

#include "testing.h"

int main(int argc, char *argv[])
{
  static const float alpha = 1.3;
  static const float beta = 2.4;

  const enum CBLAS_ORDER Order = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const enum CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans : ((atoi(argv[2])==0) ? CblasNoTrans : CblasTrans);
  const enum CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans : ((atoi(argv[3])==0) ? CblasNoTrans : CblasTrans);
  const int max_size = (argc <= 4) ? 32 : atoi(argv[4]);
  const int stride = (argc <= 5) ? 1 : atoi(argv[5]);
  const int error_check = (argc <= 6) ? 0 : atoi(argv[6]);

  //const char order = (Order == CblasColMajor) ? 'C' : 'R';
  const char transa = (TransA == CblasNoTrans) ? 'N' : 'T';
  const char transb = (TransB == CblasNoTrans) ? 'N' : 'T';

  precision *A = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *B = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *C = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *D = (precision *)memalign(256, max_size*max_size*sizeof(precision));

  srand(time(NULL));

  for (int M = stride; M <= max_size; M += stride) {
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

    for(int i = 0; i < M*K; i++) A[i] = (precision)rand()/(RAND_MAX);
    for(int i = 0; i < K*N; i++) B[i] = (precision)rand()/(RAND_MAX);
    for(int i = 0; i < M*N; i++) C[i] = (precision)rand()/(RAND_MAX);

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
    double comp_time;
    if (error_check) {
      memcpy(D, C, M*N*sizeof(*C));
      if (Order == CblasColMajor) {
        TESTBLAS_GEMM(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      } else {
        TESTBLAS_GEMM(transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
      }
      CBLAS_GEMM(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, D, ldc);
      error_check_gemm(C, D, M, N);
    } else {
      //comp_time = get_current_time();
      //if (Order == CblasColMajor) {
      //  TESTBLAS_GEMM(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
      //} else {
      //  TESTBLAS_GEMM(transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
      //}
      //comp_time = get_current_time() - comp_time;
      //printf(" : %10.6lf sec %10.5lf GFlop/s", comp_time, gflops/comp_time);
    }

    comp_time = get_current_time();
    if (Order == CblasColMajor) {
      TESTBLAS_GEMM(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
      TESTBLAS_GEMM(transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
    }
    comp_time = get_current_time() - comp_time;
    printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    fflush(stdout);
  }
  free(A); free(B); free(C); free(D);

  return 0;
  }
