void gen_quants(MPI_Comm comm, /* io */ Particle *opp, int *on, /**/ Quants *q) {
    gen::gen_rig_from_solvent(comm, q->nt, q->nv, q->htt, q->hvv, /* io */ opp, on, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst, q->pp_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void set_ids(const int ns, /**/ Solid *ss_hst, Solid *ss_dev) {
    gen::set_rig_ids(ns, /**/ ss_hst);
    if (ns) cH2D(ss_dev, ss_hst, ns);
}

void set_ids(Quants q) {
    set_ids(q.ns, q.ss_hst, q.ss);
}
