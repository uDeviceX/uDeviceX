#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NMAX 10000
double x0[NMAX], y0[NMAX], z0[NMAX];
double x[NMAX], y[NMAX], z[NMAX];

double rad(double x, double y, double z) { return sqrt(x*x + y*y + z*z); }
double pair(double x, double y, double z,
            double x0, double y0, double z0) {
    double dx, dy, dz;
    dx = x - x0; dy = y - y0; dz = z - z0;
    return dx*dx + dy*dy + dz*dz;
}

void dist0(int n, double x, double y, double z, double *x0, double *y0,  double *z0,
           /**/ int *ii) {
    int i, im;
    double d, dm;
    dm = 1e42;
    for (im = i = 0; i < n; i++) {
        d = pair(x, y, z, x0[i], y0[i], z0[i]);
        if (d < dm) { dm = d; im = i; }
    }
    *ii = im;
}

void dist(int n, double *x, double *y, double *z, double *x0, double *y0,  double *z0) {
    int i, j;
    double d;
    double dx, dy, dz;
    for (i = 0; i < n; i++) {
        dist0(n, x[i], y[i], z[i], x0, y0, z0, /**/ &j);
        dx = x[i] - x0[j];
        dy = y[i] - y0[j];
        dz = z[i] - z0[j];
        d = rad(dx, dy, dz);
        printf("%16.10e %16.10e %16.10e %16.10e\n", x[i], y[i], z[i], d);
    }
}

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
    dist(n0, x0, y0, z0, x0, y0, z0);
    for (;;) {
        read(f, /**/ &n, x, y, z);
        if (n == 0) break;
        if (n != n0) {
            fprintf(stderr, "punto2dist: n=%d != n0=%d\n", n, n0);
            exit(2);
        }
        printf("\n");
        dist(n, x, y, z, x0, y0, z0);
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
