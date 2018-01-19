enum {X, Y, Z};

static void update_r_hst(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

static void reinit_ft_hst(const int nsolid, /**/ Solid *ss) {
    for (int i = 0; i < nsolid; ++i) {
        Solid *s = ss + i;
            
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
        s->to[X] = s->to[Y] = s->to[Z] = 0;
    }
}

static void generate_hst(const Solid *ss_hst, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    int j, start;
    for (j = start = 0; j < ns; ++j) {
        update_r_hst(rr0, nps, ss_hst[j].com, ss_hst[j].e0, ss_hst[j].e1, ss_hst[j].e2, /**/ pp + start);
        start += nps;
    }
}

static void mesh2pp_hst(const Solid *ss_hst, const int ns, int nv, const float *vv, /**/ Particle *pp) {
    int i, j;
    for (j = 0; j < ns; ++j) {
        const Solid *s = ss_hst + j;
        update_r_hst(vv, nv, s->com, s->e0, s->e1, s->e2, /**/ pp + j * nv);

        for (i = 0; i < nv; ++i) {
            float *v = pp[j*nv + i].v;
            v[X] = v[Y] = v[Z] = 0;
        }
    }
}

static void gen_pp_hst(const int ns, const float *rr0_hst, const int nps, /**/ Solid *ss_hst, Particle *pp_hst) {
    generate_hst(ss_hst, ns, rr0_hst, nps, /**/ pp_hst);
    reinit_ft_hst(ns, /**/ ss_hst);
}

static void gen_ipp_hst(const Solid *ss_hst, const int ns, int nv, const float *vv, Particle *i_pp_hst) {
    mesh2pp_hst(ss_hst, ns, nv, vv, /**/ i_pp_hst);
}

static void cpy_H2D(const RigQuants *q) {
    cH2D(q->i_pp, q->i_pp_hst, q->ns * q->nv);
    cH2D(q->ss,   q->ss_hst,   q->ns);
    cH2D(q->rr0,  q->rr0_hst,  q->nps * 3);
    cH2D(q->pp,   q->pp_hst,   q->n);
}
