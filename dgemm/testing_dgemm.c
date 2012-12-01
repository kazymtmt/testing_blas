#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TESTING_ATLAS
#include <cblas.h>
#include <clapack.h>
#define DGEMM_COL() cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define DGEMM_ROW() cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#elif defined(TESTING_MKL)
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#define DGEMM_COL() cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define DGEMM_ROW() cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#elif defined(TESTING_AMDCLBLAS)
#include <cblas.h>
#include <clapack.h>
#include <clAmdBlas.h>
#define DGEMM_COL() clAmdBlasDgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)
#define DGEMM_ROW() clAmdBlasDgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)
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
  int M, N, K, lda, ldb, ldc, sizeA, sizeB, sizeC;

#if defined(TESTING_ATLAS) || defined(TESTING_AMDCLBLAS)
  const enum CBLAS_ORDER Order = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const enum CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans : ((atoi(argv[2])==0) ? CblasNoTrans : CblasTrans);
  const enum CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans : ((atoi(argv[3])==0) ? CblasNoTrans : CblasTrans);
#elif defined(TESTING_MKL)
  const CBLAS_ORDER Order = (argc <= 1) ? CblasColMajor : ((atoi(argv[1])==0) ? CblasColMajor : CblasRowMajor);
  const CBLAS_TRANSPOSE TransA = (argc <= 2) ? CblasNoTrans : ((atoi(argv[2])==0) ? CblasNoTrans : CblasTrans);
  const CBLAS_TRANSPOSE TransB = (argc <= 3) ? CblasNoTrans : ((atoi(argv[3])==0) ? CblasNoTrans : CblasTrans);
#endif

  const int max_size = (argc <= 4) ? 32 : atoi(argv[4]);
  const int stride = (argc <= 5) ? 1 : atoi(argv[5]);
  const int error_check = (argc <= 6) ? 0 : atoi(argv[6]);
  double *A  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *B  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *C  = (double *)memalign(256, max_size*max_size*sizeof(double));
  double *C2 = (double *)memalign(256, max_size*max_size*sizeof(double));

#ifdef TESTING_AMDCLBLAS
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx;
  cl_command_queue queue;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  const clAmdBlasOrder order = (Order == CblasColMajor) ? clAmdBlasColumnMajor : clAmdBlasRowMajor;
  const clAmdBlasTranspose transa = (TransA == CblasNoTrans) ? clAmdBlasNoTrans : clAmdBlasTrans;
  const clAmdBlasTranspose transb = (TransB == CblasNoTrans) ? clAmdBlasNoTrans : clAmdBlasTrans;
  /* Setup OpenCL environment. */
  err = clGetPlatformIDs(1, &platform, NULL);
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  props[1] = (cl_context_properties)platform;
  ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
  queue = clCreateCommandQueue(ctx, device, 0, &err);
  /* Setup clAmdBlas. */
  err = clAmdBlasSetup();
  if (err != CL_SUCCESS) {
    printf("clAmdBlasSetup() failed with %d\n", err);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 1;
  }
  // Create buffers
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*max_size*sizeof(double), NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*max_size*sizeof(double), NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, max_size*max_size*sizeof(double), NULL, &err);
  // Dummy
  M = N = K = lda = ldb = ldc = 512;
  if (Order == CblasColMajor) {
    DGEMM_COL();
  } else {
    DGEMM_ROW();
  }
  err = clFinish(queue);
#endif

  printf("               M    N    K\n");
  for (M = stride; M <= max_size; M += stride) {
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
    sizeA = M*K;
    sizeB = K*N;
    sizeC = M*N;

    lapackf77_dlarnv(&ione, ISEED, &sizeA, A);
    lapackf77_dlarnv(&ione, ISEED, &sizeB, B);
    lapackf77_dlarnv(&ione, ISEED, &sizeC, C);
    memcpy(C2, C, sizeC*sizeof(double));

#ifdef TESTING_AMDCLBLAS
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeA*sizeof(double), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeB*sizeof(double), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, sizeC*sizeof(double), C, 0, NULL, NULL);
#endif

    printf("dgemm %c%c%c : %4d %4d %4d",
        (Order == CblasColMajor) ? 'C' : 'R',
        (TransA == CblasNoTrans) ? 'N' : 'T',
        (TransB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);

    double flops = FLOPS_DGEMM(M, N, K);
    start = my_get_current_time();
    if (Order == CblasColMajor) {
      DGEMM_COL();
    } else {
      DGEMM_ROW();
    }
#ifdef TESTING_AMDCLBLAS
    err = clFinish(queue);
#endif
    end = my_get_current_time();
    comp_time = end - start;
    perf = flops / comp_time / 1e9;

    if (error_check) {
#ifdef TESTING_AMDCLBLAS
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(double), C, 0, NULL, NULL);
#endif
      cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C2, ldc);
      blasf77_daxpy(&sizeC, &neg_one, C, &ione, C2, &ione);
      error = lapackf77_dlange("M", &M, &N, C2, &ldc, work);
      printf(" : %10.6lf sec %12.6lf GFlop/s   %e", comp_time, perf, error);
    } else {
      printf(" : %10.6lf sec %12.6lf GFlop/s   -", comp_time, perf);
    }

    start = my_get_current_time();
    if (Order == CblasColMajor) {
      DGEMM_COL();
    } else {
      DGEMM_ROW();
    }
#ifdef TESTING_AMDCLBLAS
    err = clFinish(queue);
#endif
    end = my_get_current_time();
    comp_time = end - start;
    perf = flops / comp_time / 1e9;
    printf(" : %10.6lf sec %12.6lf GFlop/s\n", comp_time, perf);
  }
  free(A); free(B); free(C); free(C2);
#ifdef TESTING_AMDCLBLAS
  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  clAmdBlasTeardown();
#endif

  return 0;
  }
