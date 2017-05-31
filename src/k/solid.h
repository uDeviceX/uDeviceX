namespace k_solid
{
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};

#define _HD_ __host__ __device__
    
    _HD_ float dot(const float *v, const float *u) {
        return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
    }

    _HD_ void reject(/**/ float *v, float *u) {
        const float d = dot(v, u);
        v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
    }

    _HD_ float norm(const float *v) {
        return sqrt(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
    }

    _HD_ void normalize(/**/ float *v) {
        const float nrm = norm(v);
        v[X] /= nrm; v[Y] /= nrm; v[Z] /= nrm;
    }

    _HD_ void gram_schmidt(/**/ float *e0, float *e1, float *e2) {
        normalize(e0);

        reject(e1, e0);
        normalize(e1);

        reject(e2, e0);
        reject(e2, e1);
        normalize(e2);
    }

    _HD_ void rot_e(const float *om, /**/ float *e)
    {
        float omx = om[X], omy = om[Y], omz = om[Z];
        float ex = e[X], ey = e[Y], ez = e[Z];
        float vx, vy, vz;

        vx = omy*ez - omz*ey;
        vy = omz*ex - omx*ez;
        vz = omx*ey - omy*ex;

        e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
    }

    __global__ void rot_referential(const float *om, /**/ float *e0, float *e1, float *e2)
    {
        if (threadIdx.x == 0)
        {
            rot_e(om, /**/ e0);
            rot_e(om, /**/ e1);
            rot_e(om, /**/ e2);
            
            gram_schmidt(/**/ e0, e1, e2);
        }
    }
    
    __device__ void warpReduceSumf3(float *x)
    {
        for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        {
            x[X] += __shfl_down(x[X], offset);
            x[Y] += __shfl_down(x[Y], offset);
            x[Z] += __shfl_down(x[Z], offset);
        }
    }

    __global__ void add_f_to(const Particle *pp, const Force *ff, const int n, Solid *s)
    {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        
        Force    f = {0, 0, 0};
        Particle p = {0, 0, 0, 0, 0, 0};

        if (gid < n)
        {
            f = ff[gid];
            p = pp[gid];
        }

        const float dr[3] = {p.r[X] - s->com[X],
                             p.r[Y] - s->com[Y],
                             p.r[Z] - s->com[Z]};
        
        float to[3] = {dr[Y] * f.f[Z] - dr[Z] * f.f[Y],
                       dr[Z] * f.f[X] - dr[X] * f.f[Z],
                       dr[X] * f.f[Y] - dr[Y] * f.f[X]};
        
        warpReduceSumf3(f.f);
        warpReduceSumf3(to);

        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(&s->fo[X], f.f[X]);
            atomicAdd(&s->fo[Y], f.f[Y]);
            atomicAdd(&s->fo[Z], f.f[Z]);

            atomicAdd(&s->to[X], to[X]);
            atomicAdd(&s->to[Y], to[Y]);
            atomicAdd(&s->to[Z], to[Z]);
        }
    }

    __global__ void reinit_ft(const int nsolid, Solid *ss)
    {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if (gid < nsolid)
        {
            Solid *s = ss + gid;
            s->fo[X] = s->fo[Y] = s->fo[Z] = 0.f;
            s->to[X] = s->to[Y] = s->to[Z] = 0.f;
        }
    }

    __global__ void update_om_v(Solid *s)
    {
        if (threadIdx.x == 0)
        {
            const float *A = s->Iinv, *b = s->to;

            const float dom[3] = {A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z],
                                  A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z],
                                  A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z]};
            
            s->om[X] += dom[X]*dt;
            s->om[Y] += dom[Y]*dt;
            s->om[Z] += dom[Z]*dt;

            if (pin_axis)
            {
                s->om[X] = s->om[Y] = 0.f;
            }

            if (pin_com)
            {
                s->v[X] = s->v[Y] = s->v[Z] = 0.f;
            }
            else
            {
                const float sc = dt/s->mass;
                
                s->v[X] += s->fo[X] * sc;
                s->v[Y] += s->fo[Y] * sc;
                s->v[Z] += s->fo[Z] * sc;
            }
        }
    }

    __global__ void compute_velocity(const Solid *s, const int n, /**/ Particle *pp)
    {
        const int pid = threadIdx.x + blockIdx.x * blockDim.x;

        float omx = s->om[X], omy = s->om[Y], omz = s->om[Z];
        
        if (pid < n)
        {
            float *r0 = pp[pid].r, *v0 = pp[pid].v;

            const float x = r0[X]-s->com[X];
            const float y = r0[Y]-s->com[Y];
            const float z = r0[Z]-s->com[Z];
            
            v0[X] = s->v[X] + omy*z - omz*y;
            v0[Y] = s->v[Y] + omz*x - omx*z;
            v0[Z] = s->v[Z] + omx*y - omy*x;
        }
    }

    __global__ void update_com(Solid *s)
    {
        if (threadIdx.x < 3)
        s->com[threadIdx.x] += s->v[threadIdx.x]*dt;
    }

    __global__ void update_r(const float *rr0, const int n, const Solid *s, /**/ Particle *pp)
    {
        const int pid = threadIdx.x + blockIdx.x * blockDim.x;

        if (pid < n)
        {
            float *r0 = pp[pid].r;
            const float *ro = &rr0[3*pid];
            float x = ro[X], y = ro[Y], z = ro[Z];

            r0[X] = s->com[X] + x*s->e0[X] + y*s->e1[X] + z*s->e2[X];
            r0[Y] = s->com[Y] + x*s->e0[Y] + y*s->e1[Y] + z*s->e2[Y];
            r0[Z] = s->com[Z] + x*s->e0[Z] + y*s->e1[Z] + z*s->e2[Z];
        }
    }

    __global__ void update_mesh(const Solid *ss_dev, const float *vv, const int nv, /**/ Particle *pp)
    {
        const int sid = blockIdx.y; // solid Id
        const Solid *s = ss_dev + sid;

        const int i = threadIdx.x + blockIdx.x * blockDim.x;;
        const int vid = sid * nv + i;
        
        if (i < nv)
        {
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
} /* namespace k_solid */
