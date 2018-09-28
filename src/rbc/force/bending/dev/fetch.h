static __device__ StressInfo fetch_stress_info(int, StressFul_v sv) {
    StressInfo si;
    si.l0 = sv.l0;
    si.a0 = sv.a0;
    return si;
}

static __device__ StressInfo fetch_stress_info(int i, StressFree_v sv) {
    StressInfo si;
    si.l0 = sv.ll[i];
    si.a0 = sv.aa[i];
    return si;
}

static __device__ Rnd0Info fetch_rnd_info(int, int, Rnd0_v rv) {
    Rnd0Info ri;
    return ri;
}

static __device__ Rnd1Info fetch_rnd_info(int i0, int j, Rnd1_v rv) {
    /* i0: edge index; j: vertex index */
    Rnd1Info ri;
    int i1;
    i1 = rv.anti[i0];
    if (i1 > i0) j = j - i0 + i1;
    ri.r = rv.rr[j];
    return ri;
}
