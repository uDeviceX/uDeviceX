#define CODE "rig"
#define PP ".pp"
#define SS ".ss"

static void gen_code(const char *name, const char *ext, char *code) {
    strcpy(code, name);
    strcat(code, ext);
}

static void pp2rr(const Particle *pp, const int n, float *rr) {
    int i, c;
    for (i = 0; i < n; ++i)
        for ( c = 0; c < 3; ++c)
            rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(int maxp, MPI_Comm comm, const char *base, const char *name, const int id, int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst) {
    Particle *pp;
    char code[FILENAME_MAX];
    
    EMALLOC(maxp, &pp);
    gen_code(name, PP, code);
    restart_read_pp(comm, base, code, RESTART_TEMPL, nps, pp);
    pp2rr(pp, *nps, rr0_hst);
    EFREE(pp);

    gen_code(name, SS, code);
    restart_read_ss(comm, base, code, id, ns, ss_hst);
    *n = *ns * (*nps);
}

void rig_strt_quants(MPI_Comm comm, const MeshRead *mesh, const char *base, const int id, RigQuants *q) {
    gen_from_strt(q->maxp, comm, base, CODE, id, /**/ &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, mesh_read_get_vert(mesh), /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(MPI_Comm comm, const char *base, const char *name, const int nps, const float *rr0_hst) {
    Particle *pp;
    char code[FILENAME_MAX];
    EMALLOC(nps, &pp);
    rr2pp(rr0_hst, nps, pp);

    gen_code(name, PP, code);
    restart_write_pp(comm, base, code, RESTART_TEMPL, nps, pp);
    
    EFREE(pp);
}

void rig_strt_dump_templ(MPI_Comm comm, const char *base, const RigQuants *q) {
    strt_dump_templ0(comm, base, CODE, q->nps, q->rr0_hst);
}

void rig_strt_dump(MPI_Comm comm, const char *base, const int id, const RigQuants *q) {
    char code[FILENAME_MAX];
    gen_code(CODE, SS, code);
    restart_write_ss(comm, base, code, id, q->ns, q->ss_hst);
}

#undef PP
#undef SS
#undef CODE
