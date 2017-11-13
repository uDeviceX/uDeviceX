#include <stdio.h>

enum{N=8100};

int main() {
    FILE *f = fopen("colors.values", "wb");

    int cc[N], i;
    
    for (i = 0; i < N; ++i) {
        cc[i] = 1;
    }
    fwrite(cc, sizeof(int), N, f);
    
    fclose(f);
}
