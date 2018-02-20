__device__ real3 tri(RbcParams_v par, int, real3 r1, real3 r2, real3 r3, Shape0 shape, real area, real volume) {
    real l0, A0, totArea, totVolume;
    l0 = shape.a;
    A0 = shape.A;
    totArea = par.totArea;
    totVolume = par.totVolume;
    return tri0(par, r1, r2, r3,   l0, A0, totArea, totVolume,   area, volume);
}
