#ifndef _FLOPS_H
#define _FLOPS_H
/*
 * Level 3 BLAS
 */

#define FMULS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))
#define FADDS_GEMM(__m, __n, __k) ((__m) * (__n) * (__k))

#define FLOPS_DGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )
#define FLOPS_SGEMM(__m, __n, __k) (     FMULS_GEMM((double)(__m), (double)(__n), (double)(__k)) +       FADDS_GEMM((double)(__m), (double)(__n), (double)(__k)) )

#endif
