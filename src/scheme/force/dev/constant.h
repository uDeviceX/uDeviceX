__global__ void main(float mass, float ax, float ay, float az,
                     int n, const Particle*, /**/ Force *ff) {
    enum {X, Y, Z};
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;
    float *f = ff[pid].f;
    f[X] += mass * ax;
    f[Y] += mass * ay;
    f[Z] += mass * az;
}
