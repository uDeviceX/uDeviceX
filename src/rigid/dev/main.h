__global__ void rot_referential(float dt, const int ns, Rigid *ss) {
    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ns) {
        Rigid s = ss[i];
        rot_e(dt, s.om, /**/ s.e0);
        rot_e(dt, s.om, /**/ s.e1);
        rot_e(dt, s.om, /**/ s.e2);

        gram_schmidt(/**/ s.e0, s.e1, s.e2);
        ss[i] = s;
    }
}

static __device__ void atomicAdd3(float d[3], float3 s) {
    enum {X, Y, Z};
    atomicAdd(d + X, s.x);
    atomicAdd(d + Y, s.y);
    atomicAdd(d + Z, s.z);
}

static __device__ float3 fetch_force(int i, const Force *ff) {
    enum {X, Y, Z};
    float3 f;
    f.x = ff[i].f[X];
    f.y = ff[i].f[Y];
    f.z = ff[i].f[Z];
    return f;
}

__global__ void add_f_to(const int nps, const Particle *pp, const Force *ff, /**/ Rigid *ss) {
    enum {X, Y, Z};
    int gid, sid, i;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    sid = blockIdx.y;

    i = sid * nps + gid;

    float3 to, f = make_float3(0, 0, 0);
    Particle   p = {0, 0, 0, 0, 0, 0};

    if (gid < nps) {
        f = fetch_force(i, ff);
        p = pp[i];
    }

    const float dr[3] = {p.r[X] - ss[sid].com[X],
                         p.r[Y] - ss[sid].com[Y],
                         p.r[Z] - ss[sid].com[Z]};

    to.x = dr[Y] * f.z - dr[Z] * f.y;
    to.y = dr[Z] * f.x - dr[X] * f.z;
    to.z = dr[X] * f.y - dr[Y] * f.x;

    f =  warpReduceSum(f);
    to = warpReduceSum(to);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd3(ss[sid].fo, f);
        atomicAdd3(ss[sid].to, to);
    }
}

__global__ void reinit_ft(const int ns, Rigid *ss) {
    enum {X, Y, Z};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < ns) {
        Rigid *s = ss + gid;
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0.f;
        s->to[X] = s->to[Y] = s->to[Z] = 0.f;
    }
}

__global__ void update_om_v(RigPinInfo pi, float dt, const int ns, Rigid *ss) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < ns) {
        Rigid s = ss[gid];
        const float *A = s.Iinv, *b = s.to;
        const float sc = dt/s.mass;
        const float dom[3] = {A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z],
                              A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z],
                              A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z]};

        s.om[X] += dom[X]*dt;
        s.om[Y] += dom[Y]*dt;
        s.om[Z] += dom[Z]*dt;

        if (pi.axis.x) s.om[X] = 0;
        if (pi.axis.y) s.om[Y] = 0;
        if (pi.axis.z) s.om[Z] = 0;

        
        s.v[X] += s.fo[X] * sc;
        s.v[Y] += s.fo[Y] * sc;
        s.v[Z] += s.fo[Z] * sc;
        
        if (pi.com.x) s.v[X] = 0;
        if (pi.com.y) s.v[Y] = 0;
        if (pi.com.z) s.v[Z] = 0;
        
        ss[threadIdx.x] = s;
    }
}

__global__ void update_com(float dt, int ns, Rigid *ss) {
    int sid, c, i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    sid = i / 3;
    c   = i % 3;

    if (sid < ns)
        ss[sid].com[c] += ss[sid].v[c]*dt;
}

__global__ void update_pp(const int nps, const float *rr0, const Rigid *ss, /**/ Particle *pp) {
    enum {X, Y, Z};
    int pid, sid, i;
    Rigid s;
    float x, y, z, dx, dy, dz, omx, omy, omz;
    Particle p;
    const float *r0;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    sid = blockIdx.y;
    i = sid * nps + pid;

    s = ss[sid];
    omx = s.om[X]; omy = s.om[Y]; omz = s.om[Z];

    if (pid < nps) {
        p = pp[i];
        r0 = &rr0[3*pid];
        x = r0[X]; y = r0[Y]; z = r0[Z];

        dx = x * s.e0[X] + y * s.e1[X] + z * s.e2[X];
        dy = x * s.e0[Y] + y * s.e1[Y] + z * s.e2[Y];
        dz = x * s.e0[Z] + y * s.e1[Z] + z * s.e2[Z];

        p.v[X] = s.v[X] + omy * dz - omz * dy;
        p.v[Y] = s.v[Y] + omz * dx - omx * dz;
        p.v[Z] = s.v[Z] + omx * dy - omy * dx;

        p.r[X] = s.com[X] + dx;
        p.r[Y] = s.com[Y] + dy;
        p.r[Z] = s.com[Z] + dz;

        pp[i] = p;
    }
}

__global__ void update_mesh(float dt, const Rigid *ss_dev, const int nv, const float *vv, /**/ Particle *pp) {
    enum {X, Y, Z};
    const int sid = blockIdx.y; // solid Id
    const Rigid *s = ss_dev + sid;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int vid = sid * nv + i;

    if (i < nv) {
        const float x = vv[3*i + X];
        const float y = vv[3*i + Y];
        const float z = vv[3*i + Z];

        const Particle p0 = pp[vid];
        Particle p;

        p.r[X] = x * s->e0[X] + y * s->e1[X] + z * s->e2[X] + s->com[X];
        p.r[Y] = x * s->e0[Y] + y * s->e1[Y] + z * s->e2[Y] + s->com[Y];
        p.r[Z] = x * s->e0[Z] + y * s->e1[Z] + z * s->e2[Z] + s->com[Z];

        p.v[X] = (p.r[X] - p0.r[X]) / dt;
        p.v[Y] = (p.r[Y] - p0.r[Y]) / dt;
        p.v[Z] = (p.r[Z] - p0.r[Z]) / dt;

        pp[vid] = p;
    }
}
