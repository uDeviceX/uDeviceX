static __device__ float3 dpd(int dpid, float4 rdest, float4 udest, float4 rsrc, float4 usrc, int spid) {
    float rnd, fx, fy, fz;
    float3 r1, r2, v1, v2;
    rnd = rnd::mean0var1ii( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );
    f2tof3(rdest, &r1); f2tof3(rsrc, &r2);
    f2tof3(udest, &v1); f2tof3(usrc, &v2);

    forces::dpd0(SOLVENT_TYPE, SOLVENT_TYPE,
                     r1.x, r1.y, r1.z,
                     r2.x, r2.y, r2.x,
                     v1.x, v1.y, v1.z,
                     v2.x, v2.y, v2.z,
                     rnd,
                     &fx, &fy, &fz);
    return make_float3(fx, fy, fz);
}
