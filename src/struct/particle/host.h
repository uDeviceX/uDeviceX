static float3 fetch_pos(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 r;
    r.x = pp.r[X]; r.y = pp.r[Y]; r.z = pp.r[Z];
    return v;
}

static float3 fetch_vel(int i, const Particle *pp) {
    enum {X, Y, Z};
    float3 v;
    v.x = pp.v[X]; v.y = pp.v[Y]; v.z = pp.v[Z];
    return v;
}
