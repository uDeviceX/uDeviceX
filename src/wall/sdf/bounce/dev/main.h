enum {
    MAX_RESCUE = 8,
    MAX_NEWTON = 2
};

static __device__ void p2rv(const Particle *pp, int i, /**/ float3 *r, float3 *v) {
    enum {X, Y, Z};
    Particle p = pp[i];    
    r->x = p.r[X];
    r->y = p.r[Y];
    r->z = p.r[Z];

    v->x = p.v[X];
    v->y = p.v[Y];
    v->z = p.v[Z];
}

static __device__ void rv2p(float3 r, float3 v, int i, /**/ Particle *pp) {
    enum {X, Y, Z};
    Particle p = {r.x, r.y, r.z, v.x, v.y, v.z};
    pp[i] = p;
}

static __device__ bool is_small(float f) {return fabs(f) < 1e-6f;}
static __device__ void crop(float dt0, float *t) {
    if (*t < -dt0) *t = -dt0;
    if (*t >   0) *t = 0;
}

static __device__ void rescue(Wvel_v wv, Coords_v c, Sdf_v *texsdf, float currsdf, /* io */ float3 *r, float3 *v) {
    float sdf0, jump;
    float3 dsdf;
    int l;
    
    dsdf = ugrad(texsdf, r);
    sdf0 = currsdf;

    axpy(-sdf0, &dsdf, /**/ r);
        
    for (l = MAX_RESCUE; l >= 1; --l) {
        if (sdf(texsdf, r->x, r->y, r->z) < 0) {
            bounce_vel(wv, c, *r, /**/ v);
            return;
        }
        jump = 1.1f * sdf0 / (1 << l);
        axpy(-jump, &dsdf, /**/ r);
    }
}

static __device__ void bounce_back_1p(Wvel_v wv, Coords_v c, Sdf_v *texsdf, float currsdf,
                                      /* io */ float3 *r, float3 *v) {
    float3 r0, rc, rw, dsdf;
    float phi, dphi, t;
    int l;

    float dt0; dt0 = wv.dt0;
    assert(dt0 >=0.95*dt && dt0 <=1.05*dt);

    r0 = *r;
    // get previous position
    axpy(-dt0, v, /**/ &r0);

    if (sdf(texsdf, r0.x, r0.y, r0.z) >= 0) {
        rescue(wv, c, texsdf, currsdf, /* io */ r, v);
        return;
    }

    /* use Newton iterations to solve sdf(rw) = 0 */
    /* where rw = r + t * v, t in [-dt0, 0]        */
    t = 0;
    for (l = 0; l < MAX_NEWTON; ++l) {
        rc = *r;
        axpy(t, v, /**/ &rc);
        phi = sdf(texsdf, rc.x, rc.y, rc.z);
        dsdf = grad(texsdf, &rc);
        dphi = dot<float> (v, &dsdf);

        if (is_small(dphi))
            break;
        
        t -= phi/dphi;
        crop(dt0, &t);
    }

    rw = *r;
    axpy(t, v, /**/ &rw);
    bounce_vel(wv, c, rw, /**/ v);

    *r = rw;
    axpy(-t, v, /**/ r);
    
    if (sdf(texsdf, r->x, r->y, r->z) >= 0)
        *r = r0;    
}

__global__ void bounce_back(Wvel_v wv, Coords_v c, Sdf_v texsdf, int n, /**/ Particle *pp) {
    float s, currsdf;
    float3 r, v;
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p2rv(pp, i, /**/ &r, &v);

    s = cheap_sdf(&texsdf, r.x, r.y, r.z);

    if (s >= texsdf.cheap_threshold) {
        currsdf = sdf(&texsdf, r.x, r.y, r.z);
        if (currsdf >= 0) {
            bounce_back_1p(wv, c, &texsdf, currsdf, /*io*/ &r, &v);
            rv2p(r, v, i, /**/ pp);
        }
    }
}
