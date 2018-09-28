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
