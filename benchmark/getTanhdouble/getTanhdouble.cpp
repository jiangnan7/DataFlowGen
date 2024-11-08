#include <stdlib.h>
#include <iostream>


int getTanhint(int A[100]){

  int i,beta;
  int result ;
    int d = 0;
  for (i = 0; i < 100; i++){
       beta = A[i];

		if (beta < 1){
          result = ((beta*beta+19)*beta*beta+3)*beta;
        }
        else{
          // An if condition in the loop causes irregular computation.
	      // Static scheduler reserves time slot for each iteration
	      // causing unnecessary pipeline stalls.
          result = 1.0;
        }

        d += result;
  }
  return d;

}


int main(){
    int A[100];

  for (int i = 0; i < 100; i++){
       A[i] = i;
       std::cout << i << ", ";
       if(i%20 == 0) std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout<<  getTanhint(A);
}