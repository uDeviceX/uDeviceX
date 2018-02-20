__device__ real3 tri(RbcParams_v par, int nv, real3 r1, real3 r2, real3 r3, Shape0 shape0, real area, real volume) {
    int nt;
    real a0, A0, totArea, totVolume;
    nt = 2*nv - 4;
    totArea = par.totArea;
    totVolume = par.totVolume;
    A0 = totArea / nt;
    a0 = sqrt(A0 * 4.0 / sqrt(3.0));    
    return tri0(par, r1, r2, r3,   a0, A0, totArea, totVolume,   area, volume);
}
