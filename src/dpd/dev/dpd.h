__device__ float3 dpd(int dpid, float4 rdest, float4 udest, float4 rsrc, float4 usrc, int spid) {
    float rnd;
    float3 r1, r2, v1, v2;
    float3 f;

    rnd = rnd::mean0var1ii( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );
    f2tof3(rdest, &r1); f2tof3(rsrc, &r2);
    f2tof3(udest, &v1); f2tof3(usrc, &v2);

    f = forces::dpd(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd);
    return f;
}
