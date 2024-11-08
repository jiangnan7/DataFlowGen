#include <string.h>
#include <stdio.h>
#include <iostream>

int dropout_gold(int i[1024], int sel_ptr[128]) {//int o[1024], 
  int buf_sel =0, sel =0;
  int value =0;
  for (unsigned int k = 0; k < 1024; ++k) {
    if (!(k % 8))
      buf_sel = sel_ptr[k >> 3];
    sel = buf_sel & 0x01;
    // o[k] = sel ? (i[k] * 2) : 0;
    value += sel ? (i[k] * 2) : 0;
    buf_sel >>= 1;
  }
  return value;
}



int main(){

    int i[1024];
    int o[1024];
    int sel_ptr[128];
    for(int idx=0; idx < 1024; idx++){
        i[idx] = idx;
        if(idx%20 == 0) std::cout << std::endl;
        std::cout << idx << ",";
    }
    std::cout << std::endl;

    for(int idx=0; idx < 128; idx++){
        sel_ptr[idx] = idx;
        if(idx%20 == 0) std::cout << std::endl;
        std::cout << idx << ",";
    }

    int result =  dropout_gold(i, sel_ptr);
     std::cout << result << ",";
}
