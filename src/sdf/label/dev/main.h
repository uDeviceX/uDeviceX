static __global__ void main(const sdf::tex3Dca texsdf, int n, const Particle *pp, /**/ int *labels) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    Particle p = pp[pid];
    float sdf0 = sdf::sub::dev::sdf(texsdf, p.r[X], p.r[Y], p.r[Z]);
    labels[pid] = (int)(sdf0 >= 0) + (int)(sdf0 > 2);
}
