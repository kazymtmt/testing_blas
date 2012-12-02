#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(TESTING_ATLAS) || defined(TESTING_AMDCLBLAS) || defined(TESTING_ACML) || defined(TESTING_CUBLAS) || defined(TESTING_MAGMA)
#include <cblas.h>
#include <clapack.h>
#endif

#if defined(TESTING_ATLAS)
#define SGEMM_COL() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

#elif defined(TESTING_MKL)
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#define SGEMM_COL() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)

#elif defined(TESTING_AMDCLBLAS)
#include <clAmdBlas.h>
#define SGEMM_COL() clAmdBlasSgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)
#define SGEMM_ROW() clAmdBlasSgemm(order, transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc, 1, &queue, 0, NULL, &event)

#elif defined(TESTING_ACML)
#include <acml.h>
#define SGEMM_COL() sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
#define SGEMM_ROW() sgemm(transb, transa, N, K, K, alpha, B, ldb, A, lda, beta, C, ldc)

#elif defined(TESTING_CUBLAS)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define SGEMM_COL() cublasSgemm(handle, transa, transb, M, N, K, &alpha, bufA, lda, bufB, ldb, &beta, bufC, ldc);
#define SGEMM_ROW() cublasSgemm(handle, transb, transa, N, M, K, &alpha, bufB, ldb, bufA, lda, &beta, bufC, ldc);

#elif defined(TESTING_MAGMA)
#include <magma.h>
#define SGEMM_COL() magmablas_sgemm(transa, transb, M, N, K, alpha, bufA, lda, bufB, ldb, beta, bufC, ldc);
#define SGEMM_ROW() magmablas_sgemm(transb, transa, N, M, K, alpha, bufB, ldb, bufA, lda, beta, bufC, ldc);

#endif

#include "flops.h"
#include "testing.h"

int main(int argc, char *argv[])
{
  static const float alpha =  0.29;
  static const float beta  = -0.48;
  int ione = 1;
  int ISEED[4] = {1,0,0,1};
  float neg_one = S_NEG_ONE;
  double comp_time, perf, error;
  float work[1];
  int M, N, K, lda, ldb, ldc, sizeA, sizeB, sizeC;
#if defined(TESTING_CUBLAS)
   cudaEvent_t start, end;
   float comp_time_f;
#else
   double start, end;
#endif

#if defined(TESTING_ATLAS) || defined(TESTING_AMDCLBLAS) || defined(TESTING_ACML) || defined(TESTING_CUBLAS) || defined(TESTING_MAGMA)
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
  float *A, *B, *C, *C2;
#ifdef TESTING_MAGMA
  magma_malloc_cpu((void**) &A,  max_size*max_size*sizeof(float));
  magma_malloc_cpu((void**) &B,  max_size*max_size*sizeof(float));
  magma_malloc_cpu((void**) &C,  max_size*max_size*sizeof(float));
  magma_malloc_cpu((void**) &C2, max_size*max_size*sizeof(float));
#else
  A  = (float *)memalign(256, max_size*max_size*sizeof(float));
  B  = (float *)memalign(256, max_size*max_size*sizeof(float));
  C  = (float *)memalign(256, max_size*max_size*sizeof(float));
  C2 = (float *)memalign(256, max_size*max_size*sizeof(float));
#endif

#if defined(TESTING_AMDCLBLAS)
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
  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*max_size*sizeof(*A), NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  max_size*max_size*sizeof(*B), NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, max_size*max_size*sizeof(*C), NULL, &err);
  // Dummy
  M = N = K = lda = ldb = ldc = 512;
  if (Order == CblasColMajor) {
    SGEMM_COL();
  } else {
    SGEMM_ROW();
  }
  err = clFinish(queue);

#elif defined(TESTING_CUBLAS)
  const cublasOperation_t transa = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t transb = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  //cublasStatus stat;
  float *bufA, *bufB, *bufC;
  // Initialize CUBLAS
  cublasHandle_t handle;
  cudaStream_t stream;
  cublasCreate(&handle);
  cudaStreamCreate(&stream);
  cublasSetStream(handle, stream);
  cudaEventCreate(&start);
  cudaEventCreate(&end);

#elif defined(TESTING_MAGMA)
  const char transa = (TransA == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;
  const char transb = (TransB == CblasNoTrans) ? MagmaNoTrans : MagmaTrans;
  //cublasStatus stat;
  float *bufA, *bufB, *bufC;
  // Initialize CUBLAS
  if(CUBLAS_STATUS_SUCCESS != cublasInit()) {
    fprintf(stderr, "cublasInit(): failed\n");
    exit(1);
  }

#elif defined(TESTING_ACML)
  const char transa = (TransA == CblasNoTrans) ? 'N' : 'T';
  const char transb = (TransB == CblasNoTrans) ? 'N' : 'T';
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

    lapackf77_slarnv(&ione, ISEED, &sizeA, A);
    lapackf77_slarnv(&ione, ISEED, &sizeB, B);
    lapackf77_slarnv(&ione, ISEED, &sizeC, C);
    memcpy(C2, C, sizeC*sizeof(*C));

#if defined(TESTING_AMDCLBLAS)
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeA*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeB*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, sizeC*sizeof(*C), C, 0, NULL, NULL);
#elif defined(TESTING_CUBLAS)
    // Prepare memory objects
    cudaMalloc((void**)(&bufA), sizeA*sizeof(*A));
    cudaMalloc((void**)(&bufB), sizeB*sizeof(*B));
    cudaMalloc((void**)(&bufC), sizeC*sizeof(*C));
    cudaMemcpy(bufA, A, sizeA*sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(bufB, B, sizeB*sizeof(*B), cudaMemcpyHostToDevice);
    cudaMemcpy(bufC, C, sizeC*sizeof(*C), cudaMemcpyHostToDevice);
#elif defined(TESTING_MAGMA)
    magma_malloc((void**)(&bufA), sizeA*sizeof(*A));
    magma_malloc((void**)(&bufB), sizeB*sizeof(*B));
    magma_malloc((void**)(&bufC), sizeC*sizeof(*C));
    magma_ssetmatrix(M, K, A, lda, bufA, lda);
    magma_ssetmatrix(K, N, B, ldb, bufB, ldb);
    magma_ssetmatrix(M, N, C, ldc, bufC, ldc);
#endif

    printf("sgemm %c%c%c : %4d %4d %4d",
        (Order == CblasColMajor) ? 'C' : 'R',
        (TransA == CblasNoTrans) ? 'N' : 'T',
        (TransB == CblasNoTrans) ? 'N' : 'T',
        M, N, K);

    double flops = FLOPS_SGEMM(M, N, K);
#if defined(TESTING_CUBLAS)
    cudaEventRecord(start, stream);
#else
    start = my_get_current_time();
#endif
    if (Order == CblasColMajor) {
      SGEMM_COL();
    } else {
      SGEMM_ROW();
    }
#if defined(TESTING_AMDCLBLAS)
    err = clFinish(queue);
#endif
#if defined(TESTING_CUBLAS)
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&comp_time_f, start, end);
    comp_time = comp_time_f / 1e3;
#else
    end = my_get_current_time();
    comp_time = end - start;
#endif
    perf = flops / comp_time / 1e9;

    if (error_check) {
#if defined(TESTING_AMDCLBLAS)
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeC*sizeof(*C), C, 0, NULL, NULL);
#elif defined(TESTING_CUBLAS)
      cudaMemcpy(C, bufC, sizeC*sizeof(*C), cudaMemcpyDeviceToHost);
#elif defined(TESTING_MAGMA)
      magma_sgetmatrix(M, N, bufC, ldc, C, ldc);
#endif
      cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C2, ldc);
      blasf77_saxpy(&sizeC, &neg_one, C, &ione, C2, &ione);
#if defined(TESTING_ACML)
      error = lapackf77_slange("M", &M, &N, C2, &ldc, work, ione);
#else
      error = lapackf77_slange("M", &M, &N, C2, &ldc, work);
#endif
      printf(" : %10.6lf sec %12.6lf GFlop/s   %e", comp_time, perf, error);
    } else {
      printf(" : %10.6lf sec %12.6lf GFlop/s   -", comp_time, perf);
    }

#if defined(TESTING_CUBLAS)
    cudaEventRecord(start, stream);
#else
    start = my_get_current_time();
#endif
    if (Order == CblasColMajor) {
      SGEMM_COL();
    } else {
      SGEMM_ROW();
    }
#ifdef TESTING_AMDCLBLAS
    err = clFinish(queue);
#endif
#if defined(TESTING_CUBLAS)
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&comp_time_f, start, end);
    comp_time = comp_time_f / 1e3;
#else
    end = my_get_current_time();
    comp_time = end - start;
#endif
    perf = flops / comp_time / 1e9;
    printf(" : %10.6lf sec %12.6lf GFlop/s\n", comp_time, perf);
    fflush(stdout);

#if defined(TESTING_CUBLAS)
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(bufC);
#elif defined(TESTING_MAGMA)
    magma_free(bufA);
    magma_free(bufB);
    magma_free(bufC);
#endif
  }
  free(A); free(B); free(C); free(C2);
#if defined(TESTING_AMDCLBLAS)
  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  clAmdBlasTeardown();
#elif defined(TESTING_CUBLAS)
  cudaEventDestroy(end);
  cudaEventDestroy(start);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
#elif defined(TESTING_MAGMA)
  cublasShutdown();
#endif

  return 0;
}
