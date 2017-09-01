static __device__ float3 dpd(int dpid, float4 rdest, float4 udest, float4 rsrc, float4 usrc, int spid) {
    enum {X, Y, Z};
    float rnd, fx, fy, fz;
    float r1[3], r2[3], v1[3], v2[3];
    forces::Pa p;
    rnd = rnd::mean0var1ii( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );
    f4tof3(rdest, r1); f4tof3(rsrc, r2);
    f4tof3(udest, v1); f4tof3(usrc, v2);

    forces::dpd0(SOLVENT_TYPE, SOLVENT_TYPE,
                     r1[X], r1[Y], r1[Z],
                     r2[X], r2[Y], r2[Z],
                     v1[X], v1[Y], v1[Z],
                     v2[X], v2[Y], v2[Z],
                     rnd,
                     &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}
