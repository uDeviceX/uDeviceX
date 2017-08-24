#include <stdio.h>

void g(int dw, int a[]) {
    if (dw % 2 == 0) {
        a[0] = dw / 2;
        a[1] = 16 + dw / 2;
    } else {
        a[0] = 16 + dw / 2;
        a[1] = dw / 2;
    }
}

int main(int argc, char **argv) {
    int dw, a[2];
    while (scanf("%d", &dw) == 1) {
        g(dw, a);
        printf("%d %d %d\n", dw, a[0], a[1]);
    }
}
