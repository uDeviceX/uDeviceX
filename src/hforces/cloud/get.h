namespace hforces { namespace dev {
__device__ void cloudA_get(CloudA c, int i, /**/ forces::Pa *p) {
    float *pp, *r, *v;
    pp = c.pp;
    r = &pp[6*i];
    v = &pp[6*i + 3];
    forces::rvk2p(r, v, SOLVENT_TYPE, /**/ p);
}

__device__ void cloudB_get(CloudB c, int i, /**/ forces::Pa *p) {
    float2 *pp;
    float2 *d0, *d1, *d2;
    pp = &c.pp[3*i];

    d0 = pp++; d1 = pp++; d2 = pp;
    forces::f2k2p(*d0, *d1, *d2, SOLVENT_TYPE, /**/ p);
}
}} /* namespace */
