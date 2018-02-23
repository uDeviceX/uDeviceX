#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double disc(double a, double b, double c) { return b*b - a*c; }
double sgn(double x) {
    if (x > 0)       return  1;
    else if (x == 0) return  0;
    else             return -1;
}

void qdrtc(double A, double B, double C, /**/ double *pX1, double *pY1, double *pX2, double *pY2) {
    double b, q, r, X1, Y1, X2, Y2;
    b = -B/2;
    q = disc(A, b, C);
    if (q < 0) {
        X1 = b/A;
        X2 = X1;
        Y1 = sqrt(-q)/A;
        Y2 = - Y1;
    } else {
        Y1 = 0; Y2 = 0;
        r = b + sgn(b)*sqrt(q);
        if (r == 0) {
            X1 = C/A; X2 = -X1;
        } else {
            X1 = C/r; X2 = r/A;
        }
    }
    *pX1 = X1; *pY1 = Y1; *pX2 = X2; *pY2 = Y2;
}

void  eeval(double X, double A, double B, double C, double D, /**/
            double *pQ, double *pdQ, double *pB1, double *pC2) {
    double q0, B1, C2, dQ, Q;
    q0 = A*X;
    B1 = q0 + B;
    C2 = B1*X + C;
    dQ = (q0 + B1)*X + C2;
    Q  = C2*X + D;
    *pQ = Q; *pdQ = dQ; *pB1 = B1; *pC2 = C2;
}

void qbc(double A, double B, double C, double D,
         /**/ double *pX, double *pX1, double *pY1, double *pX2, double *pY2) {
    double X1, Y1, X2, Y2;
    qdrtc(A, B, C, /**/ &X1, &Y1, &X2, &Y2);
    fprintf(stderr, "%g %g  %g %g\n", X1, Y1, X2, Y2);
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
    if (c != 5) {
        fprintf(stderr, "needs four args\n");
        exit(2);
    }
    i = 1;
    A = read_real(v[i++]);
    B = read_real(v[i++]);
    C = read_real(v[i++]);
    D = read_real(v[i++]);
    qbc(A, B, C, D, &X, &X1, &Y1, &X2, &Y2);
}
