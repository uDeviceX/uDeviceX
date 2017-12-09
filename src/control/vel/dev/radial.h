static __device__ float3 transform(Coords c, const Particle p) {
    enum {X, Y, Z};
    float3 u; // radial coordinates
    float x, y, r, rinv;
    float cost, sint;
    x = p.r[X] - glb::r0[X];
    y = p.r[Y] - glb::r0[Y];

    r = sqrt(x*x + y*y);
    rinv = 1 / r;
    cost = rinv * x;
    sint = rinv * y;
    
    u.x = r * (  cost * p.v[X] + sint * p.v[Y]);
    u.y = r * (- sint * p.v[X] + cost * p.v[Y]);
    u.z = p.v[Z];
    return u;
}
