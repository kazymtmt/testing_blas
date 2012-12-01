#ifndef TESTING_H_

#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#define D_NEG_ONE (-1)

double my_get_current_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

#ifndef FORTRAN_NAME
#if defined(ADD_)
#define FORTRAN_NAME(lcname, UCNAME)  lcname##_
#elif defined(NOCHANGE)
#define FORTRAN_NAME(lcname, UCNAME)  lcname
#elif defined(UPCASE)
#define FORTRAN_NAME(lcname, UCNAME)  UCNAME
#else
#error Define one of ADD_, NOCHANGE, or UPCASE for how Fortran functions are name mangled.
#endif
#endif

#define blasf77_daxpy      FORTRAN_NAME( daxpy,  DAXPY  )
#define lapackf77_dlange   FORTRAN_NAME( dlange, DLANGE )
#define lapackf77_dlarnv   FORTRAN_NAME( dlarnv, DLARNV )

void blasf77_daxpy(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
double lapackf77_dlange(const char *norm, const int *m, const int *n, const double *A, const int *lda, double *work);
void lapackf77_dlarnv(const int *idist, int *iseed, const int *n, double *x);

#endif // ifndef TESTING_H_

