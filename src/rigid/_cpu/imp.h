namespace rig {

void reinit_ft(const int nsolid, /**/ Solid *ss) {
    reinit_ft_hst(nsolid, /**/ ss);
}

void update(const Force *ff, const float *rr0, int n, int nsolid, /**/ Particle *pp, Solid *shst) {
    int start = 0;
    const int nps = n / nsolid; /* number of particles per solid */
        
    for (int i = 0; i < nsolid; ++i) {
        dev::update_1s(ff + start, rr0, nps, /**/ pp + start, shst + i);
        start += nps;
    }
}

void generate(const Solid *ss, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    generate_hst(ss, ns, rr0, nps, /**/ pp);
}

void update_mesh(const Solid *ss_hst, const int ns, const Mesh m, /**/ Particle *pp) {
    for (int j = 0; j < ns; ++j) {
        const Solid *s = ss_hst + j;
                        
        for (int i = 0; i < m.nv; ++i) {
            const float* ro = m.vv + 3*i;
            const Particle p0 = pp[j * m.nv + i];
            float *r = pp[j * m.nv + i].r;
            float *v = pp[j * m.nv + i].v;
                
            const float x = ro[X], y = ro[Y], z = ro[Z];
            r[X] = x * s->e0[X] + y * s->e1[X] + z * s->e2[X] + s->com[X];
            r[Y] = x * s->e0[Y] + y * s->e1[Y] + z * s->e2[Y] + s->com[Y];
            r[Z] = x * s->e0[Z] + y * s->e1[Z] + z * s->e2[Z] + s->com[Z];
                
            v[X] = (r[X] - p0.r[X]) / dt;
            v[Y] = (r[Y] - p0.r[Y]) / dt;
            v[Z] = (r[Z] - p0.r[Z]) / dt;
        }
    }
}

} // rig
