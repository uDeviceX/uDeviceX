namespace rig {
namespace sub {
namespace ic {

void ini(const char *fname, const Mesh m, /**/ int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp)
{
    int npsolid = 0;
    float3 minbb, maxbb;
    float *coms = new float[MAX_SOLIDS * 3 * 10];
        
    int nsolid = read_coms(fname, coms);
    
    if (nsolid == 0) ERR("No solid provided.\n");
    
    mesh::get_bbox(m.vv, m.nv, /**/ &minbb, &maxbb);
        
    nsolid = duplicate_PBC(minbb, maxbb, nsolid, /**/ coms);

    make_local(nsolid, coms);

    const int npp0 = *s_n;
    int *tags = new int[npp0];
    int *rcounts = new int[nsolid];

    count_pp_inside(s_pp, *s_n, coms, nsolid, m.tt, m.vv, m.nt, /**/ tags, rcounts);

    int root, idmax;
    elect(rcounts, nsolid, /**/ &root, &idmax);
    
    MC(MPI_Bcast(&idmax, 1, MPI_INT, root, l::m::cart));

    int rcount = 0;

    kill(idmax, tags, /**/ s_n, s_pp, &rcount, r_pp);

    delete[] rcounts;
    delete[] tags;

    share_parts(root, /**/ r_pp, &rcount);

    Solid model;

    // share model to everyone
    
    if (m::rank == root)
    {
        npsolid = rcount;
        if (!npsolid) ERR("No particles remaining in root node.\n");
            
        for (int d = 0; d < 3; ++d)
        model.com[d] = coms[idmax*3 + d];
    
        solid::ini(r_pp, npsolid, solid_mass, model.com, m, /**/ rr0, &model);

        empty_solid(m, /* io */ rr0, &npsolid);
    }
    
    MC(MPI_Bcast(&npsolid,       1,   MPI_INT, root, l::m::cart) );
    MC(MPI_Bcast(rr0,  3 * npsolid, MPI_FLOAT, root, l::m::cart) );
    MC(MPI_Bcast(&model, 1, datatype::solid, root, l::m::cart) );
    
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

    set_ids(nsolid, /**/ ss);

    delete[] coms;
}

} // ic
} // rig
} // sub
