#include <stdio.h>

enum{
    N = 8100,
    T = 200
};

int main() {
    FILE *f = fopen("colors.values", "wb");

    int cc[N], i;
    
    for (i = 0; i < N; ++i) {
        cc[i] = i > T ? 1 : 0;
    }
    fwrite(cc, sizeof(int), N, f);
    
    fclose(f);
}
