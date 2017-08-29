template <typename T>
static void remove(T *data, int nv, int *e, int nc) {
    int c; /* c: cell */
    for (c = 0; c < nc; c++) cA2A(data + nv*c, data + nv*e[c], nv);
}

static void remove_rbcs(rbc::Quants *q, sdf::Quants qsdf) {
    int stay[MAX_CELL_NUM];
    int nc0;
    q->nc = sdf::who_stays(qsdf, q->pp, q->n, nc0 = q->nc, q->nv, /**/ stay);
    q->n = q->nc * q->nv;
    remove(q->pp, q->nv, stay, q->nc);
    MSG("%d/%d RBCs survived", q->nc, nc0);
}

void remove_solids(rig::Quants *q, sdf::Quants qsdf) {
    int stay[MAX_SOLIDS];
    int ns0;
    int nip = q->ns * q->m_dev.nv;
    q->ns = sdf::who_stays(qsdf, q->i_pp, nip, ns0 = q->ns, q->m_dev.nv, /**/ stay);
    q->n  = q->ns * q->nps;
    remove(q->pp,       q->nps,      stay, q->ns);
    remove(q->pp_hst,   q->nps,      stay, q->ns);

    remove(q->ss,       1,           stay, q->ns);
    remove(q->ss_hst,   1,           stay, q->ns);

    remove(q->i_pp,     q->m_dev.nv, stay, q->ns);
    remove(q->i_pp_hst, q->m_hst.nv, stay, q->ns);
    MSG("sim.impl: %d/%d Solids survived", q->ns, ns0);
}

