#define CODE "rig"
#define PP CODE ".pp"
#define SS CODE ".ss"

static void pp2rr(const Particle *pp, const int n, float *rr) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c)
    rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(int maxp, MPI_Comm comm, const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst) {
    Particle *pp;
    EMALLOC(maxp, &pp);
    restart_read_pp(comm, BASE_STRT_READ, PP, RESTART_TEMPL, nps, pp);
    pp2rr(pp, *nps, rr0_hst);
    EFREE(pp);

    restart_read_ss(comm, BASE_STRT_READ, SS, id, ns, ss_hst);
    *n = *ns * (*nps);
}

void rig_strt_quants(MPI_Comm comm, const int id, RigQuants *q) {
    gen_from_strt(q->maxp, comm, id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, q->hvv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(MPI_Comm comm, const int nps, const float *rr0_hst) {
    Particle *pp;
    EMALLOC(nps, &pp);
    rr2pp(rr0_hst, nps, pp);

    restart_write_pp(comm, BASE_STRT_DUMP, PP, RESTART_TEMPL, nps, pp);
    
    EFREE(pp);
}

void rig_strt_dump_templ(MPI_Comm comm, const RigQuants *q) {
    strt_dump_templ0(comm, q->nps, q->rr0_hst);
}

void rig_strt_dump(MPI_Comm comm, const int id, const RigQuants *q) {
    restart_write_ss(comm, BASE_STRT_DUMP, SS, id, q->ns, q->ss_hst);
}

#undef PP
#undef SS
#undef CODE
