int matrixadd(int array_0[100], int array_1[100], int array_2[100], int array_3[100]) {

  int s_0 = 0 ;
  int s_1 = 0 ;
  int s_2 = 0 ;
  int s_3 = 0 ;

loop_0:

  for (int i = 0; i < 100; i++) {
    int temp = array_0[i];
    if (temp != 0 )
      s_0 += temp;
  }
loop_1:

  for (int i = 0; i < 100; i++) {
    int temp = array_1[i];
    if (temp != 0 )
      s_1 += temp;
  }
loop_2:

  for (int i = 0; i < 100; i++) {
    int temp = array_2[i];
    if (temp != 0 )
      s_2 += temp;
  }

loop_3:

  for (int i = 0; i < 100; i++) {
    int temp = array_3[i];
    if (temp != 0 )
      s_3 += temp;
  }

  return s_0 + s_1 + s_2 + s_3  ;

}