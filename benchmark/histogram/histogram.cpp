#include "histogram.h"
#include<iostream>

#include "histogram.h"
#include <stdlib.h>

#define N 32768
#define BIN_MAX_NUM 20

#include "histogram.h"

void histogram(int f[100], int  w[100], int  hist[100]) 
{
  for (int i=0; i<100; ++i) {
	  int  temp = w[i];
	int  x = hist[f[i]];
	hist[f[i]] +=  temp;
	}
}

int main(){
	int f[100];
	int  w[100], hist[100];
	
	for(int i = 0; i<100; i++){

		f[i] = i;
		w[i] = 1;
		hist[i] = 0;
	}

	histogram(f, w, hist);

	int t = 0;
  for (int i= 0; i<100; i++)
		t+=(hist[i]==1);


  for(int i = 0; i<100; i++){
		std::cout << hist[i] << ", ";
		if(i%20 == 0) std::cout << std::endl;
	}

for(int i = 0; i<100; i++){
		std::cout << f[i] << ", ";
		if(i%20 == 0) std::cout << std::endl;
	}

 for(int i = 0; i<100; i++){
		std::cout <<  "0, ";
		if(i%20 == 0) std::cout << std::endl;
	}

  std::cout << std::endl << "t: " << t;
  if (t ==100)
  return 0;
    else
  return 1;

}