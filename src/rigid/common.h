namespace rig {

enum {X, Y, Z};
enum {XX, XY, XZ, YY, YZ, ZZ};
enum {YX = XY, ZX = XZ, ZY = YZ};

#define _HD_ __host__ __device__

void update_r_hst(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp) {
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

void reinit_ft_hst(const int nsolid, /**/ Solid *ss) {
    for (int i = 0; i < nsolid; ++i) {
        Solid *s = ss + i;
            
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
        s->to[X] = s->to[Y] = s->to[Z] = 0;
    }
}

void generate_hst(const Solid *ss_hst, const int ns, const float *rr0, const int nps, /**/ Particle *pp) {
    int start = 0;
    for (int j = 0; j < ns; ++j) {
        update_r_hst(rr0, nps, ss_hst[j].com, ss_hst[j].e0, ss_hst[j].e1, ss_hst[j].e2, /**/ pp + start);
        start += nps;
    }
}


} // rig
