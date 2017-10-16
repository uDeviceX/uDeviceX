static void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Solid *ss_hst, Particle *pp_hst) {
    rig::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    rig::reinit_ft_hst(ns, /**/ ss_hst);
}

static void gen_ipp_hst(const Solid *ss_hst, const int ns, int nv, const float *vv, Particle *i_pp_hst) {
    rig::mesh2pp_hst(ss_hst, ns, nv, vv, /**/ i_pp_hst);
}

static void cpy_H2D(const Quants *q) {
    cH2D(q->i_pp, q->i_pp_hst, q->ns * q->nv);
    cH2D(q->ss,   q->ss_hst,   q->ns);
    cH2D(q->rr0,  q->rr0_hst,  q->nps * 3);
    cH2D(q->pp,   q->pp_hst,   q->n);
}
