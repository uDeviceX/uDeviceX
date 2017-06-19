void load_solid_mesh(const char *fname, Mesh *m_dev, Mesh *m_hst) {
    l::ply::read(fname, /**/ m_hst);

    m_dev->nv = m_hst->nv;
    m_dev->nt = m_hst->nt;

    CC(cudaMalloc(&m_dev->tt, 3 * m_dev->nt * sizeof(int)));
    CC(cudaMalloc(&m_dev->vv, 3 * m_dev->nv * sizeof(float)));

    cH2D(m_dev->tt, m_hst->tt, 3 * m_dev->nt);
    cH2D(m_dev->vv, m_hst->vv, 3 * m_dev->nv);
}

void create(Particle *opp, int *on) {
    load_solid_mesh("mesh_solid.ply");

    // generate models
    MSG("start solid ini");
    ic::ini("ic_solid.txt", m_hst, /**/ &ns, &nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG("done solid ini");

    allocate_tcells();
    
    // generate the solid particles

    solid::generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    solid::reinit_ft_hst(ns, /**/ ss_hst);
    npp = ns * nps;

    solid::mesh2pp_hst(ss_hst, ns, m_hst, /**/ i_pp_hst);
    cH2D(i_pp_dev, i_pp_hst, ns * m_hst.nv);

    cH2D(ss_dev, ss_hst, ns);
    cH2D(rr0, rr0_hst, 3 * nps);

    cH2D(pp, pp_hst, npp);

    MC(l::m::Barrier(m::cart));
}
