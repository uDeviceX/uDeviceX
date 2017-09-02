namespace hforces { namespace dev {
__device__ void cloudA_get(CloudA c, int i, /**/ forces::Pa *p) {
    float *pp, *r, *v;
    pp = c.pp;
    r = &pp[6*i];
    v = &pp[6*i + 3];
    rvk2p(r, v, SOLVENT_TYPE, /**/ p);
}

void cloudB_get(CloudB c, int i, /**/ forces::Pa *p) {
}
}} /* namespace */
