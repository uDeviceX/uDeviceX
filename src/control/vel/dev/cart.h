static __device__ float3 transform(const Particle p) {
    enum {X, Y, Z};
    float3 u;
    u.x = p.u[X];
    u.y = p.u[Y];
    u.z = p.u[Z];
    return u;
}
