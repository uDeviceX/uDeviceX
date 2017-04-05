namespace k_solid
{
#define X 0
#define Y 1
#define Z 2
#define XX 0
#define XY 1
#define XZ 2
#define YY 3
#define YZ 4
#define ZZ 5

#define YX XY
#define ZX XZ
#define ZY YZ

#define _HD_ __host__ __device__

        
    _HD_ float dot(float *v, float *u) {
        return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
    }

    _HD_ void reject(/**/ float *v, float *u) {
        float d = dot(v, u);
        v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
    }

    _HD_ float norm(float *v) {
        return sqrt(v[X]*v[X]+v[Y]*v[Y]+v[Z]*v[Z]);
    }

    _HD_ void normalize(/**/ float *v) {
        float nrm = norm(v);
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

    /* wrap COM to the domain; TODO: many processes */
    _HD_ void pbc_solid(/**/ float *com)
    {
        float lo[3] = {-0.5*XS, -0.5*YS, -0.5*ZS};
        float hi[3] = { 0.5*XS,  0.5*YS,  0.5*ZS};
        float L[3] = {XS, YS, ZS};
        for (int c = 0; c < 3; ++c) {
            while (com[c] <  lo[c]) com[c] += L[c];
            while (com[c] >= hi[c]) com[c] -= L[c];
        }
    }

#if defined(rsph)

#define shape sphere
#define pin_axis (false)

    namespace sphere
    {
        _HD_ bool inside(float x, float y, float z) {
            return x*x + y*y + z*z < rsph*rsph;
        }
    }

#elif defined(rcyl)

#define shape cylinder
#define pin_axis (true)

    namespace cylinder
    {
        _HD_ bool inside(float x, float y, float z) {
            return x*x + y*y < rcyl * rcyl;
        }
    }

#elif defined(a2_ellipse)

#define shape ellipse
#define pin_axis (true)

    namespace ellipse
    {
#define a2 a2_ellipse 
#define b2 b2_ellipse
        
        _HD_ bool inside(float x, float y, float z) {
            return x*x / a2 + y*y / b2 < 1;
        }
    }

#elif defined(a2_ellipsoid)

#define shape ellipsoid
    
    namespace ellipsoid
    {
#define a2 a2_ellipsoid
#define b2 b2_ellipsoid
#define c2 c2_ellipsoid

        __DH__ bool inside(float x, float y, float z) {
            return x*x / a2 + y*y / b2 + z*z / c2 < 1;
        }
    }
#endif

    
    _HD_ bool inside(float x, float y, float z) {
        return shape::inside(x, y, z);
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

    __global__ void add_f_to(const Particle *pp, const Force *ff, const int n, const float *com, float *ftot, float *ttot)
    {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;

        Force    f = {0, 0, 0};
        Particle p = {0, 0, 0, 0, 0, 0};

        if (gid < n)
        {
            f = ff[gid];
            p = pp[gid];
        }

        const float dr[3] = {p.r[X]- com[X],
                             p.r[Y]- com[Y],
                             p.r[Z]- com[Z]};
        
        float to[3] = {dr[Y] * f.f[Z] - dr[Z] * f.f[Y],
                       dr[Z] * f.f[X] - dr[X] * f.f[Z],
                       dr[X] * f.f[Y] - dr[Y] * f.f[X]};
        
        warpReduceSumf3(f.f);
        warpReduceSumf3(to);

        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(&ftot[X], f.f[X]);
            atomicAdd(&ftot[Y], f.f[Y]);
            atomicAdd(&ftot[Z], f.f[Z]);

            atomicAdd(&ttot[X], to[X]);
            atomicAdd(&ttot[Y], to[Y]);
            atomicAdd(&ttot[Z], to[Z]);
        }
    }

    __global__ void reinit_ft(float *fo, float *to)
    {
        if (threadIdx.x == 0)
        {
            fo[X] = fo[Y] = fo[Z] = 0.f;
            to[X] = to[Y] = to[Z] = 0.f;
        }
    }

    __global__ void update_om_v(const float mass, const float *Iinv, const float *fo, const float *to,
                                /**/ float *om, float *v)
    {
        if (threadIdx.x == 0)
        {
            const float *A = Iinv, *b = to;

            const float dom[3] = {A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z],
                                  A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z],
                                  A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z]};
            
            om[X] += dom[X]*dt; om[Y] += dom[Y]*dt; om[Z] += dom[Z]*dt;

            if (pin_axis)
            {
                om[X] = om[Y] = 0.f;
            }

            if (pin_com)
            {
                v[X] = v[Y] = v[Z] = 0.f;
            }
            else
            {
                const float sc = dt/mass;
                
                v[X] += fo[X] * sc;
                v[Y] += fo[Y] * sc;
                v[Z] += fo[Z] * sc;
            }
        }
    }

    __global__ void compute_velocity(const float *v, const float *com, const float *om, const int n, /**/ Particle *pp)
    {
        const int pid = threadIdx.x + blockIdx.x * blockDim.x;

        float omx = om[X], omy = om[Y], omz = om[Z];
        
        if (pid < n)
        {
            float *r0 = pp[pid].r, *v0 = pp[pid].v;

            const float x = r0[X]-com[X];
            const float y = r0[Y]-com[Y];
            const float z = r0[Z]-com[Z];
            
            v0[X] = v[X] + omy*z - omz*y;
            v0[Y] = v[Y] + omz*x - omx*z;
            v0[Z] = v[Z] + omx*y - omy*x;
        }
    }

    __global__ void update_com(const float *v, float *com)
    {
        if (threadIdx.x == 0)
        {
            com[X] += v[X]*dt; com[Y] += v[Y]*dt; com[Z] += v[Z]*dt;
            pbc_solid(/**/ com);
        }
    }

    __global__ void update_r(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2, /**/ Particle *pp)
    {
        const int pid = threadIdx.x + blockIdx.x * blockDim.x;

        if (pid < n)
        {
            float *r0 = pp[pid].r;
            const float *ro = &rr0[3*pid];
            float x = ro[X], y = ro[Y], z = ro[Z];

            r0[X] = com[X] + x*e0[X] + y*e1[X] + z*e2[X];
            r0[Y] = com[Y] + x*e0[Y] + y*e1[Y] + z*e2[Y];
            r0[Z] = com[Z] + x*e0[Z] + y*e1[Z] + z*e2[Z];
        }
    }

#undef X
#undef Y
#undef Z
#undef XX
#undef XY
#undef XZ
#undef YY
#undef YZ
#undef ZZ

#undef YX
#undef ZX
#undef ZY
    
} /* namespace k_solid */
