void rig_gen_quants(const Coords *coords, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo *pi, MPI_Comm comm,
                    /* io */ Particle *opp, int *on, /**/ RigQuants *q) {
    RigGenInfo rgi;
    FluInfo fluinfo;
    RigInfo riginfo;
    
    rgi.mass = rig_mass;
    rgi.pi = pi;
    rgi.tt = q->htt; rgi.nt = q->nt;
    rgi.vv = q->hvv; rgi.nv = q->nv;
    rgi.empty_pp = empty_pp;
    rgi.numdensity = numdensity;

    fluinfo.pp = opp;
    fluinfo.n = on;    

    riginfo.ns = &q->ns;
    riginfo.nps = &q->nps;
    riginfo.n = &q->n;
    riginfo.rr0 = q->rr0_hst;
    riginfo.ss = q->ss_hst;
    riginfo.pp = q->pp_hst;
    
    gen::gen_rig_from_solvent(coords, comm, rgi, /* io */ fluinfo, /**/ riginfo);
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
