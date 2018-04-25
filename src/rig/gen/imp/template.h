`_S_ int3 fid2shift(int3 L, int fid) {
    int3 s;
    using namespace frag_hst;
    s.x = - i2dx(fid) * L.x;
    s.y = - i2dy(fid) * L.y;
    s.z = - i2dz(fid) * L.z;
    return s;
}

_S_ void extract_and_shift_hst(int3 s, int n, const Particle *pp, const int *labels, int *nx, float *rrx) {
    enum {X, Y, Z};
    int i, j;
    const float *r;
    float *rx;
    j = *nx;
    for (i = 0; i < n; ++i) {
        if (labels[i] == IN) {
            r = pp[i].r;
            rx = &rrx[3 * j];
            rx[X] = r[X] + s.x;
            rx[Y] = r[Y] + s.y;
            rx[Z] = r[Z] + s.z;
            ++j;
        }
    }
    *nx = j;
}

_S_ void label_extract_and_shift(int3 shift, int pdir, int n, const Particle *pp_dev, const Particle *pp_hst, int nt, int nv,
                                    int nm, const int4 *tt, const Particle *pp_mesh,
                                    /**/ int *ntempl, float *rrtempl, /*w*/ int *ll_dev, int *ll_hst) {
    UC(compute_labels(pdir, n, pp_dev, nt, nv, nm, tt, pp_mesh, IN, OUT, /**/ ll_dev));
    cD2H(ll_hst, ll_dev, n);
    extract_and_shift_hst(shift, n, pp_hst, ll_hst, ntempl, rrtempl);
}

_S_ void collect_and_broadcast_template(MPI_Comm comm, int *n, float *rr) {
    int i, rank, size, *starts, *counts, ntot;
    float *rr_recv;
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    EMALLOC(size, &counts);
    EMALLOC(size, &starts);
    
    MC(m::Allgather(n, 1, MPI_INT, counts, 1, MPI_INT, comm));

    starts[0] = 0;
    for (i = 0; i < size; ++i) counts[i] *= 3;
    for (i = 1; i < size; ++i) starts[i] = starts[i-1] + counts[i-1];
    ntot = starts[size-1] + counts[size-1];

    EMALLOC(ntot, &rr_recv);
    
    MC(m::Allgatherv(rr, counts[rank], MPI_FLOAT, rr_recv, counts, starts, MPI_INT, comm));

    *n = ntot/3;
    memcpy(rr, rr_recv, ntot * sizeof(float));
    
    EFREE(rr_recv);
    EFREE(starts);
    EFREE(counts);
}

_S_ void label_template_dev(int pdir, int3 L, MPI_Comm cart, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh,
                               int nflu, const Particle *pp_dev, const Particle *pp_hst, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int i, maxm, nmall, n, cc[NFRAGS];
    int3 shift;
    Particle *pp0, *pp;

    maxm = NFRAGS + 1;
    Dalloc(&pp, nv * maxm);
    pp0 = pp;

    nmall = nm;
    n = nm * nv;
    if (n) cD2D(pp, pp_mesh, n);
    
    UC(exchange_mesh(maxm, L, cart, nv, /* io */ &nmall, pp, /**/ cc));

    // bulk mesh
    shift = fid2shift(L, frag_bulk);
    if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
    pp += nm * nv;
    
    // halo meshes
    for (i = 0; i < NFRAGS; ++i) {
        nm = cc[i];
        shift = fid2shift(L, i);
        if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
        pp += nm * nv;
    }

    UC(collect_and_broadcast_template(cart, nps, rr0));
    
    Dfree(pp0);
}

struct Transf {
    float s[3];
    float e0[3], e1[3], e2[3];
};

_S_ void get_transf(MPI_Comm comm, bool hasid0, const Solid *ss, Transf *T) {
    int root, sz;
    if (hasid0) {
        sz = 3 * sizeof(float);
        memcpy(T->s, ss[0].com, sz);
        memcpy(T->e0, ss[0].e0, sz);
        memcpy(T->e1, ss[0].e1, sz);
        memcpy(T->e2, ss[0].e2, sz);
    }
    root = get_root(comm, hasid0);
    MC(m::Bcast(T->s, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e0, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e1, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e2, 3, MPI_FLOAT, root, comm));
}

_S_ void transform(const Transf *T, float *r) {
    enum {X, Y, Z};
    int c;
    float r0[3];
    for (c = 0; c < 3; ++c) r0[c] = r[c] - T->s[c];
    for (c = 0; c < 3; ++c)
        r[c] =
            r0[X] * T->e0[c] +
            r0[Y] * T->e1[c] +
            r0[Z] * T->e2[c];
}

_S_ void transf_template(const Transf *T, int n, float *rr) {
    int i;
    for (i = 0; i < n; ++i) transform(T, &rr[3*i]);
}

_I_ void extract_template(int3 L, MPI_Comm cart, RigGenInfo rgi, int n, const Particle *flu_pp_dev, const Particle *flu_pp_hst,
                             int ns, bool hasid0, const Solid *ss, /**/ int *nps, float *rr0, /*w*/ int *ll_dev, int *ll_hst) {
    int nm, pdir;
    Transf T;
    *nps = 0;
    nm = hasid0 ? 1 : 0;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    UC(label_template_dev(pdir, L, cart, rgi.nt, rgi.nv, nm, rgi.tt, rgi.pp, n, flu_pp_dev, flu_pp_hst, /**/ nps, rr0, /*w*/ ll_dev, ll_hst));
    UC(get_transf(cart, hasid0, ss, &T));
    UC(transf_template(&T, *nps, rr0));
}

