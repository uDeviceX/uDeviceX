namespace mdstr {
namespace sub {
namespace dev {

#define i2del(i) {((i)     + 1) % 3 - 1,        \
                  ((i) / 3 + 1) % 3 - 1,        \
                  ((i) / 9 + 1) % 3 - 1}

__global__ void shift(int n, const int fid, /**/ Particle *pp) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    int d[3] = i2del(fid);
    Particle p = pp[i];
    int L[3] = {XS, YS, ZS};
    for (int c = 0; c < 3; ++c) p.r[c] -= d[c] * L[c];
    pp[i] = p;
}

#undef i2del

} // dev
} // sub
} // mdstr
