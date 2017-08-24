#include <stdio.h>

int f0(int dw) {
    return (32 * ((dw    ) & 0x1) + dw) >> 1;
}

int f1(int dw) {
    return (32 * ((dw + 1) & 0x1) + dw) >> 1;
}

int main(int argc, char **argv) {
    int dw, rc;
    while (scanf("%d", &dw) == 1)
        printf("%d %d %d\n", dw, f0(dw), f1(dw));
}
