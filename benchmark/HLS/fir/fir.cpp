
#include <stdlib.h>
#include<iostream>

int fir (int d_i[100], int idx[100] ) {
	int i;
	int tmp=0;

	For_Loop: for (i=0;i<100;i++) {
		tmp += idx [i] * d_i[99-i];

	}

	return tmp;
}

