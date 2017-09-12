__global__ void iniP(int n, Particle *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    Particle p = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    pp[i] = p;
}

__global__ void updP(int n, Particle *pp) {
    enum {X, Y, Z};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    Particle p = pp[i];
    p.r[X] += dt * p.v[X];
    p.r[Y] += dt * p.v[Y];
    p.r[Z] += dt * p.v[Z];
    pp[i] = p;
}

__global__ void inif(int n, float *pp) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    float *p = pp + 6*i;
    *(p++) = 0.f;
    *(p++) = 1.f;
    *(p++) = 2.f;
    *(p++) = 3.f;
    *(p++) = 4.f;
    *(p++) = 5.f;
}

__global__ void updf(int n, float *pp) {
    enum {X, Y, Z};
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float *r = pp + 6*i;
    const float *v = r + 3;

    *(r++) += dt * *(v++);
    *(r++) += dt * *(v++);
    *(r++) += dt * *(v++);
}
