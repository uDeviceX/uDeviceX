#include <stdio.h>
#include <stdlib.h>

#define NMAX 100000
double x0[NMAX], y0[NMAX], z0[NMAX];
double x[NMAX], y[NMAX], z[NMAX];

void read(FILE *f, /**/ int *pn, double *x, double *y, double *z) {
    int i;
    char s[BUFSIZ];
    for (i = 0; ; i++) {
        if (fgets(s, sizeof(s) - 1, f) == '\0') break;
        if (sscanf(s, "%lf %lf %lf", &x[i], &y[i], &z[i]) != 3) break;
    }
    *pn = i;
}

void main0(FILE *f) {
    int n0, n;
    read(f, /**/ &n0, x0, y0, z0);
    for (;;) {
        read(f, /**/ &n, x, y, z);
        if (n == 0) break;
        if (n != n0) {
            fprintf(stderr, "punto2dist: n=%d != n0=%d\n", n, n0);
            exit(2);
        }
    }
}

int main(int c, const char **v) {
    FILE *f;
    const char *path;
    if (c < 2) {
        fprintf(stderr, "punto2dist: not enough args\n");
        exit(2);
    }
    path = v[1];
    if ((f = fopen(path, "r")) == NULL) {
        fprintf(stderr, "punto2dist: fail to open '%s'\n", v[0]);
        exit(2);
    }

    main0(f);
}
