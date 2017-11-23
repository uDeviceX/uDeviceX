void setn(int n)    { g::n = n; }
void setv(float *v) {
    enum {X, Y, Z};
    g::v[X] = v[X]; g::v[Y] = v[Y]; g::v[Z] = v[Z];
}
void get(/**/ int *n, float *v) { /* report statistics */
    enum {X, Y, Z};
    v[X] = g::v[X]; v[Y] = g::v[Y]; v[Z] = g::v[Z];
    *n = g::n;
}
