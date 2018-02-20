void rig_gen_quants(const Coords *coords, float rig_mass, const RigPinInfo *pi, MPI_Comm comm, /* io */ Particle *opp, int *on, /**/ RigQuants *q) {
    gen::gen_rig_from_solvent(coords, rig_mass, pi, comm, q->nt, q->nv, q->htt, q->hvv, /* io */ opp, on, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst, q->pp_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void set_ids(MPI_Comm comm, const int ns, /**/ Solid *ss_hst, Solid *ss_dev) {
    gen::set_rig_ids(comm, ns, /**/ ss_hst);
    if (ns) cH2D(ss_dev, ss_hst, ns);
}

void rig_set_ids(MPI_Comm comm, RigQuants *q) {
    set_ids(comm, q->ns, q->ss_hst, q->ss);
}
