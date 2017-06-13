void load_solid_mesh(const char *fname) {
    ply::read(fname, &s::m_hst);

    s::m_dev.nv = s::m_hst.nv;
    s::m_dev.nt = s::m_hst.nt;

    CC(cudaMalloc(&(s::m_dev.tt), 3 * s::m_dev.nt * sizeof(int)));
    CC(cudaMalloc(&(s::m_dev.vv), 3 * s::m_dev.nv * sizeof(float)));

    CC(cudaMemcpy(s::m_dev.tt, s::m_hst.tt, 3 * s::m_dev.nt * sizeof(int), H2D));
    CC(cudaMemcpy(s::m_dev.vv, s::m_hst.vv, 3 * s::m_dev.nv * sizeof(float), H2D));
}

void allocate() {
    mpDeviceMalloc(&s::pp);
    mpDeviceMalloc(&s::ff);

    CC(cudaMalloc(&s::ss_dev,    MAX_SOLIDS * sizeof(Solid)));
    CC(cudaMalloc(&s::ss_bb_dev, MAX_SOLIDS * sizeof(Solid)));

    s::ss_hst      = new Solid[MAX_SOLIDS];
    s::ss_bb_hst   = new Solid[MAX_SOLIDS];
    s::ss_dmphst   = new Solid[MAX_SOLIDS];
    s::ss_dmpbbhst = new Solid[MAX_SOLIDS];

    s::i_pp_hst    = new Particle[MAX_PART_NUM];
    s::i_pp_bb_hst = new Particle[MAX_PART_NUM];
    CC(cudaMalloc(   &s::i_pp_dev, MAX_PART_NUM * sizeof(Particle)));
    CC(cudaMalloc(&s::i_pp_bb_dev, MAX_PART_NUM * sizeof(Particle)));

    s::bboxes_hst = new float[6*MAX_SOLIDS];
    CC(cudaMalloc(&s::bboxes_dev, 6*MAX_SOLIDS * sizeof(float)));
}

void allocate_tcells() {
    s::tcs_hst = new int[XS * YS * ZS];
    s::tcc_hst = new int[XS * YS * ZS];
    s::tci_hst = new int[27 * MAX_SOLIDS * s::m_hst.nt]; // assume 1 triangle don't overlap more than 27 cells

    
    CC(cudaMalloc(&s::tcs_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tcc_dev, XS * YS * ZS * sizeof(int)));
    CC(cudaMalloc(&s::tci_dev, 27 * MAX_SOLIDS * s::m_dev.nt * sizeof(int)));
}

void deallocate() {
    delete[] s::m_hst.tt;      CC(cudaFree(s::m_dev.tt));
	delete[] s::m_hst.vv;      CC(cudaFree(s::m_dev.vv));

	delete[] s::tcs_hst;       CC(cudaFree(s::tcs_dev));
	delete[] s::tcc_hst;       CC(cudaFree(s::tcc_dev));
	delete[] s::tci_hst;       CC(cudaFree(s::tci_dev));

	delete[] s::i_pp_hst;      CC(cudaFree(s::i_pp_dev));
	delete[] s::i_pp_bb_hst;   CC(cudaFree(s::i_pp_bb_dev));

	delete[] s::bboxes_hst;    CC(cudaFree(s::bboxes_dev));
	delete[] s::ss_hst;        CC(cudaFree(s::ss_dev));
	delete[] s::ss_bb_hst;     CC(cudaFree(s::ss_bb_dev));
	delete[] s::ss_dmphst;     delete[] s::ss_dmpbbhst;
}

void create(Particle *opp, int *on) {
    load_solid_mesh("mesh_solid.ply");

    // generate models
    MSG0("start solid init");
    ic::init("ic_solid.txt", s::m_hst, /**/ &s::ns, &s::nps, s::rr0_hst, s::ss_hst, on, opp, s::pp_hst);
    MSG0("done solid init");

    allocate_tcells();
    
    // generate the solid particles

    solid::generate_hst(s::ss_hst, s::ns, s::rr0_hst, s::nps, /**/ s::pp_hst);
    solid::reinit_ft_hst(s::ns, /**/ s::ss_hst);
    s::npp = s::ns * s::nps;

    solid::mesh2pp_hst(s::ss_hst, s::ns, s::m_hst, /**/ s::i_pp_hst);
    CC(cudaMemcpy(s::i_pp_dev, s::i_pp_hst, s::ns * s::m_hst.nv * sizeof(Particle), H2D));

    CC(cudaMemcpy(s::ss_dev, s::ss_hst, s::ns * sizeof(Solid), H2D));
    CC(cudaMemcpy(s::rr0, s::rr0_hst, 3 * s::nps * sizeof(float), H2D));

    CC(cudaMemcpy(s::pp, s::pp_hst, sizeof(Particle) * s::npp, H2D));

    MC(MPI_Barrier(m::cart));
}

void ini() {
    npp = ns = nps = 0;
    allocate();
}

void fin() {
    deallocate();
}
