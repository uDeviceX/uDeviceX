#include <stdio.h>
#include <stdlib.h>

void qbc(double A, double B, double C, double D,
         /**/ double *X, double *X1, double *Y1, double *X2, double *Y2) {
    
}

double read_real(const char *s) {
    double x;
    if (sscanf(s, "%lf", &x) != 1) {
        fprintf(stderr, "not a real number '%s'\n", s);
        exit(2);
    }
    return x;
}

int main(int c, char *v[]) {
    int i;
    double A, B, C, D;
    double X, X1, Y1, X2, Y2;
    if (c != 5) fprintf(stderr, "needs four args\n");
    i = 0;
    A = read_real(v[i++]);
    B = read_real(v[i++]);
    C = read_real(v[i++]);
    D = read_real(v[i++]);
    qbc(A, B, C, D, &X, &X1, &Y1, &X2, &Y2);
}
