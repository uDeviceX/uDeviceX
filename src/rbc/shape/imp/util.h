static void diff(const float *a, const float *b, /**/ float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}

static double vabs(float *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}

static void swap(double *a, double *b) {
    double t;
    t = *a; *a = *b; *b = t;
}
static int less(double *a, double *b) { return (*a) < (*b); }
static void  sort3(double *a, double *b, double *c) {
    if (less(c, b)) swap(c, b);
    if (less(b, a)) swap(b, a);
    if (less(c, b)) swap(c, b);
}
static double area_kahan(double a, double b, double c) {
    sort3(&c, &b, &a); /* a > b > c */
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c)))/4;
}
