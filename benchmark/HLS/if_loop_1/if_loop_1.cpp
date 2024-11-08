//------------------------------------------------------------------------
// If loop
//------------------------------------------------------------------------
#include<iostream>

int if_loop_1(int a[200]) {
    int i;
    int tmp;
    int sum = 0;
    for (i = 0; i < 200; i++) {
        tmp = a[i] * 2;
        if (tmp > 10) {
            sum = tmp + sum;
        }
    }
    return sum;
}

int main(){
    int a[200];
    for(int i=0; i <200; i++){
        a[i] = i;
        std::cout << a[i] << ", ";
    }
    int ddd = if_loop_1(a);
    std::cout << "\n\n"  << ddd << "\n\n"  ;
} 