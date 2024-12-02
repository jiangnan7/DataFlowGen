#include "histogram.h"
#include<iostream>

#include "histogram.h"
#include <stdlib.h>


void histogram(int f[100], int  w[100], int  hist[100]) 
{
  for (int i=0; i<100; ++i) {
	  int  temp = w[i];
	int  x = hist[f[i]];
	hist[f[i]] +=  temp;
	}
}

