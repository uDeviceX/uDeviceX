static void gen0(MPI_Comm comm, int nt, const int4 *tt, const float *vv, int nsolid, int rcount, int idmax, int root, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, Particle *r_pp) {
    Solid model;
    // share model to everyone
    int npsolid = 0, rank;
    MC(m::Comm_rank(comm, &rank));
    if (rank == root) {
        npsolid = rcount;
        if (!npsolid) ERR("No particles remaining in root node.\n");
        for (int d = 0; d < 3; ++d)
            model.com[d] = coms[idmax*3 + d];
        ini_props(npsolid, r_pp, solid_mass, model.com, nt, tt, vv, /**/ rr0, &model);
        if (empty_solid_particles)
            empty_solid(nt, tt, vv, /* io */ rr0, &npsolid);
    }

    MC(MPI_Bcast(&npsolid,       1,   MPI_INT, root, comm) );
    MC(MPI_Bcast(rr0,  3 * npsolid, MPI_FLOAT, root, comm) );
    MC(MPI_Bcast(&model, 1, datatype::solid,   root, comm) );

    // filter coms to keep only the ones in my domain
    int id = 0;
    for (int j = 0; j < nsolid; ++j) {
        const float *com = coms + 3*j;

        if (-XS/2 <= com[X] && com[X] < XS/2 &&
            -YS/2 <= com[Y] && com[Y] < YS/2 &&
            -ZS/2 <= com[Z] && com[Z] < ZS/2 ) {
            ss[id] = model;

            for (int d = 0; d < 3; ++d)
            ss[id].com[d] = com[d];

            ++id;
        }
    }

    *ns = nsolid = id;
    *nps = npsolid;

    set_rig_ids(comm, nsolid, /**/ ss);
}

static void gen1(Coords coords, MPI_Comm comm, int nt, const int4 *tt, const float *vv, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ int *tags, int *rcounts) {
    int root, idmax;
    elect(comm, rcounts, nsolid, /**/ &root, &idmax);
    MC(MPI_Bcast(&idmax, 1, MPI_INT, root, comm));

    int rcount = 0;
    kill(idmax, tags, /**/ s_n, s_pp, &rcount, r_pp);
    DBG("after kill: %d", rcount);

    share(coords, comm, root, /**/ r_pp, &rcount);
    DBG("after share: %d", rcount);

    gen0(comm, nt, tt, vv, nsolid, rcount, idmax, root, coms, /**/ ns, nps, rr0, ss, r_pp);
}

static void gen2(Coords coords, MPI_Comm comm, int nt, const int4 *tt, const float *vv, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ int *tags, int *rcounts) {
    count_pp_inside(s_pp, *s_n, coms, nsolid, tt, vv, nt, /**/ tags, rcounts);
    gen1(coords, comm, nt, tt, vv, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ tags, rcounts);
}

static void gen3(Coords coords, MPI_Comm comm, int nt, const int4 *tt, const float *vv, int nsolid, float *coms, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp) {
    int *tags = new int[*s_n];
    int *rcounts = new int[nsolid];
    gen2(coords, comm, nt, tt, vv, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ tags, rcounts);
    delete[] rcounts;
    delete[] tags;
}

static void gen4(Coords coords, MPI_Comm comm, const char *fname, int nt, int nv, const int4 *tt, const float *vv, /**/
                 int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp,
                 /*w*/ float *coms) {
    float3 minbb, maxbb;
    int nsolid = read_coms(fname, /**/ coms);
    if (nsolid == 0) ERR("No solid provided.\n");
    mesh::get_bbox(vv, nv, /**/ &minbb, &maxbb);
    nsolid = duplicate_PBC(minbb, maxbb, nsolid, /**/ coms);
    make_local(coords, nsolid, /**/ coms);
    gen3(coords, comm, nt, tt, vv, nsolid, coms, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp);
}

void gen(Coords coords, MPI_Comm comm, const char *fname, int nt, int nv, const int4 *tt, const float *vv, /**/
         int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp) {
    float *coms = new float[MAX_SOLIDS * 3 * 10];
    gen4(coords, comm, fname, nt, nv, tt, vv, /**/ ns, nps, rr0, ss, s_n, s_pp, r_pp, /*w*/ coms);
    delete[] coms;
}

void gen_rig_from_solvent(Coords coords, MPI_Comm comm, int nt, int nv, const int4 *tt, const float *vv, /* io */ Particle *opp, int *on,
                          /* o */ int *ns, int *nps, int *n, float *rr0_hst, Solid *ss_hst, Particle *pp_hst) {
    // generate models
    MSG("start rigid gen");
    gen(coords, comm, "rigs-ic.txt", nt, nv, tt, vv, /**/ ns, nps, rr0_hst, ss_hst, on, opp, pp_hst);
    MSG("done rigid gen");

    *n = *ns * (*nps);
}
