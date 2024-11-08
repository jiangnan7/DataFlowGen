
#include <stdlib.h>

int histogram(int f[100], int  w[100], int  hist[100]) 
{ 
  int d=0;
  for (int i=0; i<100; ++i) {
	  int  temp = w[i];
    int  x = hist[f[i]];
      hist[f[i]] +=  temp;
  }
  for (int i=0; i<100; i++) {
    hist[x[i]] = hist[x[i]] + w[i];
  }
}
