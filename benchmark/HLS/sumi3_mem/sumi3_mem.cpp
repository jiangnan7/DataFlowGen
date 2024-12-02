#include <stdio.h>

int sumi3_mem(int a[200])
{
	int sum = 0;
	for (int i = 0; i < 200; i++) {
		int x = a[i];
		sum += x*x*x;
	}
	return sum;
}
