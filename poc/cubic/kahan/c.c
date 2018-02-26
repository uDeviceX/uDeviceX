#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

double max(double a, double b) { return a > b ? a : b; }
double disc(double a, double b, double c) {
    return b*b - a*c;
}
double sgn(double x) {
    return
        x  > 0 ? 1 :
        x == 0 ? 0 : -1;
}

void qdrtc(double A, double B, double C, /**/
           double *pX1, double *pY1, double *pX2, double *pY2) {
    double b, q, r, X1, Y1, X2, Y2;
    b = -B/2; q = disc(A, b, C);
    if (q < 0) {
        X1 = b/A; X2 = X1;
        Y1 = sqrt(-q)/A; Y2 = - Y1;
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
    q0 = A*X; B1 = q0 + B; C2 = B1*X + C;
    dQ = (q0 + B1)*X + C2; Q  = C2*X + D;
    *pQ = Q; *pdQ = dQ; *pB1 = B1; *pC2 = C2;
}

void qbc(double A, double B, double C, double D,
         /**/ double *pX, double *pX1, double *pY1, double *pX2, double *pY2) {
    double X, X1, Y1, X2, Y2;
    double b1, c2, q, dq, s, t, r, x0;
    if (A == 0) { X = DBL_MAX; A = B; b1 = C; c2 = D; goto fin; }
    if (D == 0) { X = 0; b1 = B; c2 = C; goto fin; }
    X =  -(B/A)/3;
    eeval(X, A, B, C, D, /**/ &q, &dq, &b1, &c2);
    t =  q/A; r =  pow(fabs(t), 1.0/3.0); s =  sgn(t);
    t =  -dq/A; if (t > 0) r = 1.324717957244746 * max(r, sqrt(t));
    x0 = X - s*r;
    if (x0 == X) goto fin;
    do {
        X = x0;
        eeval(X, A, B, C, D, /**/ &q, &dq, &b1, &c2);
        x0 = (dq == 0) ? X : X - (q/dq)/1.000000000000001;
        double d = (s * x0 - s * X);
    } while (s * x0 > s * X);
    if (fabs(A)*X*X > fabs(D/X)) {
        c2 = -D/X; b1 = (c2 - C)/X;
    }
fin:
    qdrtc(A, b1, c2, /**/ &X1, &Y1, &X2, &Y2);
    *pX = X; *pX1 = X1; *pY1 = Y1; *pX2 = X2; *pY2 = Y2;
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
    printf("%g   %g %g   %g %g\n", X, X1, Y1, X2, Y2);
}
