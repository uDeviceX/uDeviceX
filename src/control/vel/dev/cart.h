static __device__ float3 transform(Coords, TCart, const Particle p) {
    enum {X, Y, Z};
    float3 u;
    u.x = p.v[X];
    u.y = p.v[Y];
    u.z = p.v[Z];
    return u;
}
