namespace k_common {
template <typename T>
__device__ inline unsigned int fid(const T a[], const T i) {
    /* [f]ragment [id] : where is `i' in sorted a[27]? */
    unsigned int k1, k3, k9;
    k9 = 9 * ((i >= a[9])           + (i >= a[18]));
    k3 = 3 * ((i >= a[k9 + 3])      + (i >= a[k9 + 6]));
    k1 =      (i >= a[k9 + k3 + 1]) + (i >= a[k9 + k3 + 2]);
    return k9 + k3 + k1;
}

static __device__ int box(const float r[3]) {
    /* which neighboring point belongs to? */
    enum {X, Y, Z};
    int c;
    int vc[3]; /* vcode */
    int   L[3] = {XS, YS, ZS};
    for (c = 0; c < 3; ++c)
        vc[c] = (2 + (r[c] >= -L[c]/2) + (r[c] >= L[c]/2)) % 3;
    return vc[X] + 3 * (vc[Y] + 3 * vc[Z]);
}

} /* namespace */

