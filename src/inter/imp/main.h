template <typename T>
static void remove(T *data, int nv, int *e, int nc) {
    int c; /* c: cell */
    for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}

static void remove_rbcs(rbc::Quants *q, sdf::Quants qsdf) {
    int stay[MAX_CELL_NUM];
    int nc0;
    q->nc = sdf::who_stays(&qsdf, q->pp, q->n, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    MSG("%d/%d RBCs survived", q->nc, nc0);
}

static void create_solids(MPI_Comm cart, flu::Quants* qflu, rig::Quants* qrig) {
    cD2H(qflu->pp_hst, qflu->pp, qflu->n);
    rig::gen_quants(cart, /*io*/ qflu->pp_hst, &qflu->n, /**/ qrig);
    MC(m::Barrier(cart));
    cH2D(qflu->pp, qflu->pp_hst, qflu->n);
    MC(m::Barrier(cart));
    MSG("created %d solids.", qrig->ns);
}

static void remove_solids(rig::Quants *q, sdf::Quants qsdf) {
    int stay[MAX_SOLIDS];
    int ns0;
    int nip = q->ns * q->nv;
    q->ns = sdf::who_stays(&qsdf, q->i_pp, nip, ns0 = q->ns, q->nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->nv, stay, q->ns);
    remove(q->i_pp_hst, q->nv, stay, q->ns);
    MSG("sim.impl: %d/%d Solids survived", q->ns, ns0);
}

void create_walls(int maxn, sdf::Quants qsdf, flu::Quants* qflu, wall::Quants *qwall) {
    int nold = qflu->n;
    UC(wall::gen_quants(maxn, &qsdf, /**/ &qflu->n, qflu->pp, qwall));
    flu::build_cells(qflu);
    MSG("solvent particles survived: %d/%d", qflu->n, nold);
}

void freeze(MPI_Comm cart, sdf::Quants qsdf, flu::Quants *qflu, rig::Quants *qrig, rbc::Quants *qrbc) {
    MC(m::Barrier(cart));
    if (solids)           create_solids(cart, qflu, qrig);
    if (walls && rbcs  )  remove_rbcs(qrbc, qsdf);
    if (walls && solids)  remove_solids(qrig, qsdf);
    if (solids)           rig::set_ids(cart, *qrig);
}

void color_hst(Particle *pp, int n, /**/ int *cc) {
    color(pp, n, /**/ cc);
}

void color_dev(Particle *pp, int n, /*o*/ int *cc, /*w*/ Particle *pp_hst, int *cc_hst) {
    cD2H(pp_hst, pp, n);
    color(pp_hst, n, /**/ cc_hst);
    cH2D(cc, cc_hst, n);
}
