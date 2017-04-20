namespace ic_solid
{
    enum {X, Y, Z};

    int read_coms(const char * fname, /**/ float* coms)
    {
        const int L[3] = {XS, YS, ZS};
        int mi[3], nsolids = 0;
        for (int c = 0; c < 3; ++c) mi[c] = (m::coords[c] + 0.5) * L[c];
    
        if (m::rank == 0)
        {
            FILE *f = fopen("ic_solid.txt", "r"); 

            if (f == NULL)
            {
                fprintf(stderr, "Could not open ic_solid.txt. aborting.\n");
                exit(1);
            }
    
            float x, y, z;
            int i = 0;
        
            while (fscanf(f, "%f %f %f\n", &x, &y, &z) == 3)
            {
                coms[3*i + X] = x;
                coms[3*i + Y] = y;
                coms[3*i + Z] = z;

                i++;
                assert(i < MAX_SOLIDS);
            }
            nsolids = i;
        }

        MC( MPI_Bcast(&nsolids, 1,     MPI_INT,   0, m::cart) );
        MC( MPI_Bcast(coms, 3*nsolids, MPI_FLOAT, 0, m::cart) );

        // place coms in local coordinates
        for (int j = 0; j < nsolids; ++j)
        for (int d = 0; d < 3; ++d)
        coms[3*j +d] -= mi[d];
    
        return nsolids;
    }


    void count_pp_inside(const Particle *s_pp, const int n, const float *coms, const int ns, /**/ int *rcounts)
    {
        for (int ip = 0; ip < n; ++ip)
        {
            Particle p = s_pp[ip]; float *r0 = p.r;

            bool is_solid = false;
            for (int j = 0; j < ns; ++j)
            {
                const float *com = coms + 3*j;

                if (solid::inside(r0[X]-com[X], r0[Y]-com[Y], r0[Z]-com[Z]))
                {
                    assert(!is_solid);
                    ++rcounts[j];
                    is_solid = true;
                }
            }
        }

        if (m::rank == 0)
        {
            for (int j = 0; j < ns; ++j)
            printf("Found %d particles in solid %d\n", rcounts[j], j);
        }

    }

    void elect(const int *rcounts, const int ns, /**/ int *root, int *idmax)
    {
        int localmax[2] = {0, m::rank}, globalmax[2] = {0, m::rank}, idmax_ = 0;

        for (int j = 0; j < ns; ++j)
        if (localmax[0] < rcounts[j])
        {
            localmax[0] = rcounts[j];
            idmax_ = j;
        }

        delete[] rcounts;
    
        MPI_Allreduce(localmax, globalmax, 1, MPI_2INT, MPI_MAXLOC, m::cart);

        *root = globalmax[1];
        *idmax = idmax_;
    }

    void kill(const float *coms, const int ns, const int idmax, /**/ int *s_n, Particle *s_pp, int *r_n, Particle *r_pp)
    {
        int scount = 0, rcount = 0;
        
        for (int ip = 0; ip < *s_n; ++ip)
        {
            Particle p = s_pp[ip]; float *r0 = p.r;

            bool is_solid = false;
            for (int j = 0; j < ns; ++j)
            {
                const float *com = coms + 3*j;

                if (solid::inside(r0[X]-com[X], r0[Y]-com[Y], r0[Z]-com[Z]))
                {
                    if (j == idmax)
                    r_pp[rcount++] = p;
                    is_solid = true;
                }
            }
        
            if (!is_solid)
            s_pp[scount++] = p;
        }
    
        *s_n = scount;
        *r_n = rcount;
    }
    
    void init(const char * fname, /**/ int *ns, int *nps, float *rr0, Solid *ss, int *s_n, Particle *s_pp, Particle *r_pp)
    {
        float coms[MAX_SOLIDS * 3];
        
        int nsolid = read_coms(fname, coms);
        int npsolid = 0;

        if (nsolid == 0)
        {
            fprintf(stderr, "No solid provided. Aborting...\n");
            exit (1);
        }

        int *rcounts = new int[nsolid];
        for (int j = 0; j < nsolid; ++j) rcounts[j] = 0;

        count_pp_inside(s_pp, *s_n, coms, nsolid, /**/ rcounts);

        int root, idmax;
        elect(rcounts, nsolid, /**/ &root, &idmax);

        MC( MPI_Bcast(&idmax, 1, MPI_INT, root, m::cart) );

        int rcount = 0;
        
        kill(coms, nsolid, idmax, /**/ s_n, s_pp, &rcount, r_pp);

        // TODO share particles with root
        // For now I assume the solid is fully contained in domain of root.
        // TODO take Periodic BC into account
        
        Solid model;

        // share model to everyone
    
        if (m::rank == root)
        {
            npsolid = rcount;
            
            for (int d = 0; d < 3; ++d)
            model.com[d] = coms[idmax*3 + d];
    
            solid::init(r_pp, npsolid, rbc_mass, model.com, /**/ rr0, model.Iinv, model.e0, model.e1, model.e2, model.v, model.om);
        }

        MC( MPI_Bcast(&npsolid,       1,   MPI_INT, root, m::cart) );
        MC( MPI_Bcast(rr0,  3 * npsolid, MPI_FLOAT, root, m::cart) );
        MC( MPI_Bcast(&model, 1, Solid::datatype(), root, m::cart) );
        
        // filter coms to keep only the ones in my domain
        int id = 0;
        for (int j = 0; j < nsolid; ++j)
        {
            const float *com = coms + 3*j;

            if (-XS/2 <= com[X] && com[X] < XS/2 &&
                -YS/2 <= com[Y] && com[Y] < YS/2 &&
                -ZS/2 <= com[Z] && com[Z] < ZS/2 )
            {
                ss[id] = model;
            
                for (int d = 0; d < 3; ++d)
                ss[id].com[d] = com[d];

                ++id;
            }
        }

        *ns = nsolid = id;
        *nps = npsolid;
        
        // set global ids

        id = 0;
        MC( MPI_EXScan(&nsolid, &id, 1, MPI_INT, MPI_SUM, m::cart) );

        for (int j = 0; j < nsolid; ++j)
        ss[j].id = id++;
    }
}
