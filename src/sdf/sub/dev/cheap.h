namespace sdf { namespace sub { namespace dev {
static __device__ int iround(float x) {
    return (x > 0.5) ? (x + 0.5) : (x - 0.5);
}

/* within the rescaled texel width error */
inline __device__ float cheap_sdf(const tex3Dca texsdf, float x, float y, float z)  {
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM};
    int T[3] = {XTE, YTE, ZTE};
    int tc[3];
    float r[3] = {x, y, z};
    for (int c = 0; c < 3; ++c)
        tc[c] = iround(T[c] * (r[c] + L[c] / 2 + M[c]) / (L[c] + 2 * M[c]));
    return texsdf.fetch(tc[0], tc[1], tc[2]);
}

}}} /* namespace */
