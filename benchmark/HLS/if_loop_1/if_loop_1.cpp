//------------------------------------------------------------------------
// If loop
//------------------------------------------------------------------------
#include<iostream>

int if_loop_1(int a[100]) {
    int i;
    int tmp;
    int sum = 0;
    for (i = 0; i < 100; i++) {
        tmp = a[i] * 2;
        if (tmp > 10) {
            sum = tmp + sum;
        }
    }
    return sum;
}

int main(){
    int a[100];
    for(int i=0; i <100; i++){
        a[i] = i;
        std::cout << a[i] << ", ";
    }
    int ddd = if_loop_1(a);
    std::cout << "\n\n"  << ddd << "\n\n"  ;
} 
