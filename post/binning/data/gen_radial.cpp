#include <stdio.h>
#include <stdlib.h>

const int N = 4000;

void header() {
    FILE *f = fopen("rad.bop", "w");
    fprintf(f, "%d\n"
            "DATA_FILE: rad.values\n"
            "DATA_FORMAT: float\n"
            "VARIABLES: x y z u v w\n",
            N);
    fclose(f);
}

void body() {
    FILE *f = fopen("rad.values", "wb");
    int i, j;
    float x, y, z;
    srand(0);

    float pp[6*N];
    
    for (i = j = 0; i < N; ++i) {
        x = drand48();
        y = drand48();
        z = drand48();
        pp[j++] = x;
        pp[j++] = y;
        pp[j++] = z;

        pp[j++] = y-0.5;
        pp[j++] = -x+0.5;
        pp[j++] = 0;
    }

    fwrite(pp, sizeof(float), 6*N, f);
    
    fclose(f);
}

int main() {
    header();
    body();
}
