#include <stdio.h>
void set(int *a, int *n) {
    int x, i;
    i = 0;
    while (scanf("%d", &x) == 1)
        a[i++] = x;
    *n = i;
}
