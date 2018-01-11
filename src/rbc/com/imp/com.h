static void reini(int nm, /**/ float3 *h) {
    if (nm) CC(d::MemsetAsync(h, 0, nm * sizeof(float3)));
}

static void reduce(int nm, int nv, const Particle *pp, /**/ float3 *rr, float3 *vv) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nm);
    
    KL(dev::reduce_props, (blck, thrd), (nv, pp, /**/ rr, vv));
}

static void download(int nm, const float3 *d, /**/ float3 *h) {
    if (nm) CC(d::MemcpyAsync(h, d, nm * sizeof(float3), D2H));
}

static void normalize(int nm, int nv, /**/ float3 *h) {
    float fac = 1.f / nv;
    for (int i = 0; i < nm; ++i)
        scal(fac, h + i);
}

void get(int nm, int nv, const Particle *pp, /**/ ComProps *com) {
    reini(nm, /**/ com->drr);
    reini(nm, /**/ com->dvv);
    
    reduce(nm, nv, pp, /**/ com->drr, com->dvv);

    download(nm, com->drr, /**/ com->hrr);
    download(nm, com->dvv, /**/ com->hvv);
    dSync();
    
    normalize(nm, nv, /**/ com->hrr);
    normalize(nm, nv, /**/ com->hvv);
}
