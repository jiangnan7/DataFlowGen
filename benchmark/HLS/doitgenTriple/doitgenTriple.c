#include <stdlib.h>

#define N 16

void doitgenTriple(int A[N], 
                   int sum[N],int w[N*N]) {
  int p = 0;

loop_0:
  for (int i = 0; i < N; i++) {
    int s = 0;
  loop_1:
    for (int j = 0; j < N; j++) {
      int a = A[j];
      int wt = w[p + j];
      if (a > 0.0) {
        int b = a * wt;
        int c = b + wt;
        int d = c * a;
        s = s + d;
      }
    }
    p += N;
    sum[i] = s;
  }
}

