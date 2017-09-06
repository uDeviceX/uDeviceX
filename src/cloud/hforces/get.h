namespace hforces { namespace dev {
inline __device__ void cloud_get(Cloud c, int i, /**/ forces::Pa *p) {
    float *pp, *r, *v;
    pp = c.pp;
    r = &pp[6*i];
    v = &pp[6*i + 3];
    forces::rvk2p(r, v, SOLVENT_KIND, /**/ p);

    if (multi_solvent) p->color = c.cc[i];
}
}} /* namespace */
