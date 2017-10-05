static void reini(int nm, /**/ float3 *rr) {
    CC(d::MemsetAsync(rr, 0, nm * sizeof(float3)));
}

static void reduce(int nm, int nv, const Particle *pp, /**/ float3 *rr) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nm);
    
    KL(dev::reduce_position, (blck, thrd), (nv, pp, /**/ rr));
}

static void download(int nm, const float3 *drr, /**/ float3 *hrr) {
    CC(d::MemcpyAsync(hrr, drr, nm * sizeof(float3), D2H));
}

static void normalize(int nm, int nv, /**/ float3 *hrr) {
    float fac = 1.f / nv;
    for (int i = 0; i < nm; ++i)
        scal(fac, hrr + i);
}

void get_com(int nm, int nv, const Particle *pp, /**/ ComHelper *com) {
    reini(nm, /**/ com->drr);
    reduce(nm, nv, pp, /**/ com->drr);
    download(nm, com->drr, /**/ com->hrr);
    normalize(nm, nv, /**/ com->hrr);
}
