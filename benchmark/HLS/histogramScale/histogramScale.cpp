#include "histogramScale.h"
#include <iostream>

#include "histogramScale.h"
#include <stdlib.h>

void histogramScale(int f[1024], int w[1024], int hist[1024]) {
  for (int i = 0; i < 1024; ++i) {
    int temp = w[i];
    int x = hist[f[i]];
    hist[f[i]] += temp;
  }
}
