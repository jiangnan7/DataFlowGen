
#include <stdlib.h>
#include<iostream>

int fir (int d_i[100], int idx[100] ) {
	int i;
	int tmp=0;

	For_Loop: for (i=0;i<100;i++) {
		tmp += idx [i] * d_i[99-i];

	}

        //out [0] = tmp;
	return tmp;
}

int main(void){
    int d_i[100];
    int idx[100];
    int out[100];	

    for(int j = 0; j < 100; ++j){
        d_i[j] = j;
        idx[j] = j;
        std::cout << j << ", ";
        if(j%20 == 0) std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << fir(d_i, idx ) << "  res ";
}



