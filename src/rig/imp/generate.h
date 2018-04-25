static Solid gen_from_matrix(const double *A) {
    enum {X, Y, Z, W, M};
    enum {D = W};
    int c;
    Solid s; memset(&s, 0, sizeof(s));
    for (c = 0; c < D; ++c) {
        s.com[c] = A[M * c + W];
        s.e0[c]  = A[M * c + X];
        s.e1[c]  = A[M * c + Y];
        s.e2[c]  = A[M * c + Z];
    }
    return s;
}

static void gen_from_matrices(Matrices *matrices, Solid *ss) {
    int i, n;
    double *A;
    n = matrices_get_n(matrices);

    for (i = 0; i < n; ++i) {
        matrices_get(matrices, i, &A);
        ss[i] = gen_from_matrix(A);
    }
}

static void shift(const Coords *c, int n, Solid *ss) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; ++i) {
        r = ss[i].com;
        r[X] = xg2xl(c, r[X]);
        r[Y] = yg2yl(c, r[Y]);
        r[Z] = zg2zl(c, r[Z]);
    }
}

void rig_gen_mesh(const Coords *coords, MPI_Comm comm, const MeshRead *mesh, const char *ic, /**/ RigQuants *q) {
    const float *vv;
    int n, nm, nv;
    Matrices *matrices;
    nv = mesh_read_get_nv(mesh);
    vv = mesh_read_get_vert(mesh);

    UC(matrices_read_filter(ic, coords, /**/ &matrices));

    q->ns = nm = matrices_get_n(matrices);

    UC(mesh_gen_from_matrices(nv, vv, matrices, /**/ &n, q->i_pp_hst));
    UC(mesh_shift(coords, n, q->i_pp_hst));

    UC(gen_from_matrices(matrices, q->ss_hst));
    UC(shift(coords, nm, q->ss_hst));

    if (n)  cH2D(q->i_pp, q->i_pp_hst, n);
    if (nm) cH2D(q->ss, q->ss_hst, nm);
    
    UC(matrices_fin(matrices));
}

void rig_gen_quants(const Coords *coords, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo *pi, MPI_Comm comm,
                    const MeshRead *mesh, /* io */ Particle *opp, int *on, /**/ RigQuants *q) {
    RigGenInfo rgi;
    FluInfo fluinfo;
    RigInfo riginfo;
    
    rgi.mass = rig_mass;
    rgi.pi = pi;
    rgi.tt = mesh_read_get_tri(mesh);
    rgi.nt = q->nt;
    rgi.vv = mesh_read_get_vert(mesh);
    rgi.nv = q->nv;
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
    
    inter_gen_rig_from_solvent(coords, comm, rgi, /* io */ fluinfo, /**/ riginfo);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps, /**/ q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, rgi.vv, /**/ q->i_pp_hst);
    cpy_H2D(q);
}

static void set_ids(MPI_Comm comm, const int ns, /**/ Solid *ss_hst, Solid *ss_dev) {
    inter_set_rig_ids(comm, ns, /**/ ss_hst);
    if (ns) cH2D(ss_dev, ss_hst, ns);
}

void rig_set_ids(MPI_Comm comm, RigQuants *q) {
    set_ids(comm, q->ns, q->ss_hst, q->ss);
}
