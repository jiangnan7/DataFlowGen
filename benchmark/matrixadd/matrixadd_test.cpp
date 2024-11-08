#include <stdlib.h>
#include<iostream>
int matrixadd(int array_0[50], int array_1[50], int array_2[50], int array_3[50]) {

  int s_0 = 0 ;
  int s_1 = 0 ;
  int s_2 = 0 ;
  int s_3 = 0 ;

loop_0:

  for (int i = 0; i < 50; i++) {
    int temp = array_0[i];
    if (temp != 0 )
      s_0 += temp;
  }
loop_1:

  for (int i = 0; i < 50; i++) {
    int temp = array_1[i];
    if (temp != 0 )
      s_1 += temp;
  }
loop_2:

  for (int i = 0; i < 50; i++) {
    int temp = array_2[i];
    if (temp != 0 )
      s_2 += temp;
  }

loop_3:

  for (int i = 0; i < 50; i++) {
    int temp = array_3[i];
    if (temp != 0 )
      s_3 += temp;
  }

  return s_0 + s_1 + s_2 + s_3  ;

}



int main() {

  int array_0[50], array_1[50], array_2[50], array_3[50];

  for (int i = 0; i < 50; i++) {

    array_0[i] = i  ;

    array_1[i] = i ;

    array_2[i] = i ;

    array_3[i] = i ;
    
    std::cout << i << ", ";
    if(i%20 == 0) std::cout << std::endl;
  }

  int res = matrixadd(array_0, array_1, array_2, array_3 );
  std::cout << std::endl;
  std::cout << res << "  res ";
  return 0;

}