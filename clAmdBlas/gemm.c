#ifdef TESTING_DOUBLE_PRECISION
typedef double precision;
#define CLBLAS_GEMM clAmdBlasDgemm
#define CBLAS_GEMM cblas_dgemm
#else
typedef float precision;
#define CLBLAS_GEMM clAmdBlasSgemm
#define CBLAS_GEMM cblas_sgemm
#endif

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <time.h>

#include <cblas.h>
#include <clAmdBlas.h>

double get_current_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

static void error_check_gemm(const precision *C, const precision *D, const int m, const int n)
{
  double accu = 0.0;
  for (int i = 0; i < m*n; i++) {
    //double dx = fabs((double)C[i]-D[i]);
    double dx = fabs(((double)C[i]-D[i])/D[i]);
    //printf("[%d %d] %e (%f %f)\n", i/n, i%n, dx, C[i], D[i]);
    if (dx > accu) {
      accu = dx;
    }
  }
  printf(" : %e [%s]", accu, (accu < 1e-4) ? "PASSED" : "FAILED");
}

int main(int argc, char *argv[])
{
  static const cl_float alpha = 1.3;
  static const cl_float beta = 2.4;

  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
  cl_context ctx;
  cl_command_queue queue;
  cl_mem bufA, bufB, bufC;
  cl_event event = NULL;
  int ret = 0;

  clAmdBlasOrder order = (argc <= 1) ? clAmdBlasColumnMajor: ((atoi(argv[1])==0) ? clAmdBlasColumnMajor : clAmdBlasRowMajor);
  clAmdBlasTranspose transa = (argc <= 2) ? clAmdBlasNoTrans : ((atoi(argv[2])==0) ? clAmdBlasNoTrans : clAmdBlasTrans);
  clAmdBlasTranspose transb = (argc <= 3) ? clAmdBlasNoTrans : ((atoi(argv[3])==0) ? clAmdBlasNoTrans : clAmdBlasTrans);
  int max_size = (argc <= 4) ? 32 : atoi(argv[4]);
  int stride = (argc <= 5) ? 1 : atoi(argv[5]);
  int error_check = (argc <= 6) ? 0 : atoi(argv[6]);

  enum CBLAS_ORDER Order = (order == clAmdBlasColumnMajor) ? CblasColMajor : CblasRowMajor;
  enum CBLAS_TRANSPOSE TransA = (transa == clAmdBlasNoTrans) ? CblasNoTrans : CblasTrans;
  enum CBLAS_TRANSPOSE TransB = (transb == clAmdBlasNoTrans) ? CblasNoTrans : CblasTrans;

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

  precision *A = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *B = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *C = (precision *)memalign(256, max_size*max_size*sizeof(precision));
  precision *D = NULL;

  bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, max_size*max_size * sizeof(*A), NULL, &err);
  bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, max_size*max_size * sizeof(*B), NULL, &err);
  bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, max_size*max_size * sizeof(*C), NULL, &err);
  srand(time(NULL));

  // Dummy
  CLBLAS_GEMM(order, transa, transb, 1, 1, 1, alpha, bufA,
      1, bufB, 1, beta, bufC, 1, 1, &queue,
      0, NULL, &event);
  err = clFinish(queue);

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

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
        
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
      D = (precision *)memalign(256, M*N*sizeof(precision));
      memcpy(D, C, M*N*sizeof(*C));
      err = CLBLAS_GEMM(order, transa, transb, M, N, K, alpha, bufA,
          lda, bufB, ldb, beta, bufC, ldc, 1, &queue,
          0, NULL, &event);
      if (err != CL_SUCCESS) {
        printf("CLBLAS_GEMM() failed with %d\n", err);
        return 1;
      }
      //err = clWaitForEvents(1, &event);
      err = clFinish(queue);
      err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
          M * N * sizeof(*C),
          C, 0, NULL, NULL);
      CBLAS_GEMM(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, D, ldc);
      error_check_gemm(C, D, M, N);
      free(D);
    } else {
      //comp_time = get_current_time();
      //CLBLAS_GEMM(order, transa, transb, M, N, K, alpha, bufA,
      //        lda, bufB, ldb, beta, bufC, ldc, 1, &queue,
      //        0, NULL, &event);
      //err = clFinish(queue);
      //comp_time = get_current_time() - comp_time;
      //printf(" : %10.6lf sec %10.5lf GFlop/s", comp_time, gflops/comp_time);
    }

    comp_time = get_current_time();
    CLBLAS_GEMM(order, transa, transb, M, N, K, alpha, bufA,
        lda, bufB, ldb, beta, bufC, ldc, 1, &queue,
        0, NULL, &event);
    err = clFinish(queue);
    comp_time = get_current_time() - comp_time;
    printf(" : %10.6lf sec %10.5lf GFlop/s\n", comp_time, gflops/comp_time);
    fflush(stdout);
  }
  free(A); free(B); free(C);
  clReleaseMemObject(bufC);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufA);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);
  clAmdBlasTeardown();

  return ret;
  }