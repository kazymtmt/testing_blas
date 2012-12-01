#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <clapack.h>

#ifdef TESTING_ATLAS
#define DGEMM_COL() cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define DGEMM_ROW() cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#endif

#include "flops.h"
#include "testing.h"

int main(int argc, char *argv[])
{
  static const double alpha =  0.29;
  static const double beta  = -0.48;
  int ione = 1;
  int ISEED[4] = {1,0,0,1};
  double neg_one = D_NEG_ONE;
  double start, end, comp_time, perf, error, work[1];

  const enum CBLAS_ORDER order = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const enum CBLAS_TRANSPOSE transA = (argc <= 2) ? CblasNoTrans : ((atoi(argv[2])==0) ? CblasNoTrans : CblasTrans);
  const enum CBLAS_TRANSPOSE transB = (argc <= 3) ? CblasNoTrans : ((atoi(argv[3])==0) ? CblasNoTrans : CblasTrans);
  const int max_size = (argc <= 4) ? 32 : atoi(argv[4]);
  const int stride = (argc <= 5) ? 1 : atoi(argv[5]);
  const int error_check = (argc <= 6) ? 0 : atoi(argv[6]);

  double *A  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *B  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *C  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *C2 = (double *)memalign(256, max_size*max_size*sizeof(double));

  printf("               M    N    K\n");
  for (int M = stride; M <= max_size; M += stride) {
    int N, K, lda, ldb, ldc, sizeA, sizeB, sizeC;
    N = K = M;
    //K = 128;
    if (order == CblasColMajor) {
      lda = (transA == CblasTrans) ? K : M;
      ldb = (transB == CblasTrans) ? N : K;
      ldc = M;
    } else {
      lda = (transA == CblasTrans) ? M : K;;
      ldb = (transB == CblasTrans) ? K : N;
      ldc = N;
    }
    sizeA = M*K;
    sizeB = K*N;
    sizeC = M*N;

    lapackf77_dlarnv(&ione, ISEED, &sizeA, A);
    lapackf77_dlarnv(&ione, ISEED, &sizeB, B);
    lapackf77_dlarnv(&ione, ISEED, &sizeC, C);
    memcpy(C2, C, sizeC*sizeof(double));

    printf("dgemm %c%c%c : %4d %4d %4d",
        (order == CblasColMajor) ? 'C' : 'R',
        (transA == CblasNoTrans) ? 'N' : 'T',
        (transB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);

    double flops = FLOPS_DGEMM(M, N, K);
    start = my_get_current_time();
    if (order == CblasColMajor) {
      DGEMM_COL();
    } else {
      DGEMM_ROW();
    }
    end = my_get_current_time();
    comp_time = end - start;
    perf = flops / comp_time / 1e9;

    if (error_check) {
      cblas_dgemm(order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C2, ldc);
      blasf77_daxpy(&sizeC, &neg_one, C, &ione, C2, &ione);
      error = lapackf77_dlange("M", &M, &N, C2, &ldc, work);
      printf(" : %10.6lf sec %12.6lf GFlop/s   %e", comp_time, perf, error);
    } else {
      printf(" : %10.6lf sec %12.6lf GFlop/s   -", comp_time, perf);
    }

    start = my_get_current_time();
    if (order == CblasColMajor) {
      DGEMM_COL();
    } else {
      DGEMM_ROW();
    }
    end = my_get_current_time();
    comp_time = end - start;
    perf = flops / comp_time / 1e9;
    printf(" : %10.6lf sec %12.6lf GFlop/s\n", comp_time, perf);
  }
  free(A); free(B); free(C); free(C2);

  return 0;
  }
