#ifndef TESTING_H_

#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

double my_get_current_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void error_check_gemm(const precision *C, const precision *D, const int m, const int n)
{
  double accu = 0.0;
  int i;
  for (i = 0; i < m*n; i++) {
    //double dx = fabs((double)C[i]-D[i]);
    double dx = fabs(((double)C[i]-D[i])/D[i]);
    //printf("[%d %d] %e (%f %f)\n", i/n, i%n, dx, C[i], D[i]);
    if (dx > accu) {
      accu = dx;
    }
  }
  printf(" : %e [%s]", accu, (accu < 1e-4) ? "PASSED" : "FAILED");
}

#endif // ifndef TESTING_H_

