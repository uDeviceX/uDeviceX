_S_ void init_I_from_pos(int n, const float *rr, float pmass, /**/ float *I) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    int c, i;
    const float *r;
    float x, y, z;
    
    for (c = 0; c < 6; ++c) I[c] = 0;
    for (i = 0; i < n; ++i) {
        r = &rr[3*i];
        x = r[X]; y = r[Y]; z = r[Z];
        I[XX] += y*y + z*z;
        I[YY] += z*z + x*x;
        I[ZZ] += x*x + y*y;
        I[XY] -= x*y;
        I[XZ] -= z*x;
        I[YZ] -= y*z;
    }
    for (c = 0; c < 6; ++c) I[c] *= pmass;
}

_S_ void init_I_from_mesh(float density, int nt, const int4 *tt, const float *vv, /**/ float *I) {
    float com[3] = {0};
    mesh_center_of_mass(nt, tt, vv, /**/ com);
    mesh_inertia_tensor(nt, tt, vv, com, density, /**/ I);
}

_S_ void compute_properties(const RigPinInfo *pi, int n, const float *rr0, float pmass,
                               float numdensity, const MeshRead *mesh, /**/ Rigid *s) {
    enum {X, Y, Z};
    int spdir, nt;
    const int4 *tt;
    const float *vv;
    float I[6], rho;

    spdir = rig_pininfo_get_pdir(pi);

    /* ini inertia tensor */    

    if (spdir == NOT_PERIODIC) {
        rho = pmass * numdensity;
        nt = mesh_read_get_nt(mesh);
        tt = mesh_read_get_tri(mesh);
        vv = mesh_read_get_vert(mesh);
        init_I_from_mesh(rho, nt, tt, vv, /**/ I);
        s->mass = rho * mesh_volume0(nt, tt, vv);
    }
    else {
        init_I_from_pos(n, rr0, pmass, /**/ I);
        s->mass = n * pmass;
    }

    UC(linal_inv3x3(I, /**/ s->Iinv));
}

_S_ void copy_props(const Rigid *s0, Rigid *s) {
    s->mass = s0->mass;
    memcpy(s->Iinv, s0->Iinv, 6*sizeof(float));
}

_S_ void clear_vel(Rigid *s) {
    enum {X, Y, Z};
    s->v[X] = s->v[Y] = s->v[Z] = 0; 
    s->om[X] = s->om[Y] = s->om[Z] = 0; 
}

_S_ void clear_forces(Rigid *s) {
    enum {X, Y, Z};
    s->fo[X] = s->fo[Y] = s->fo[Z] = 0; 
    s->to[X] = s->to[Y] = s->to[Z] = 0; 
}

_I_ void set_properties(MPI_Comm comm, RigGenInfo rgi, int n, const float *rr0, int ns, const int *ids, /**/ Rigid *ss) {
    Rigid s_props, *s;
    int i;
    
    compute_properties(rgi.pi, n, rr0, rgi.mass, rgi.numdensity, rgi.mesh, &s_props);

    for (i = 0; i < ns; ++i) {
        s = &ss[i];
        copy_props(&s_props, s);
        clear_vel(s);
        clear_forces(s);
        s->id = ids[i];
    }
}
