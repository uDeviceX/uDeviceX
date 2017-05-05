namespace k_mesh
{
    #define _HD_ __host__ __device__

    enum {X, Y, Z};

    static _HD_ bool same_side(const float *x, const float *p, const float *a, const float *b, const float *inplane)
    {
        const float n[3] = {a[Y] * b[Z] - a[Z] * b[Y],
                            a[Z] * b[X] - a[X] * b[Z],
                            a[X] * b[Y] - a[Y] * b[X]};

        const float ndx = n[X] * (x[X] - inplane[X]) + n[Y] * (x[Y] - inplane[Y]) + n[Z] * (x[Z] - inplane[Z]);
        const float ndp = n[X] * (p[X] - inplane[X]) + n[Y] * (p[Y] - inplane[Y]) + n[Z] * (p[Z] - inplane[Z]);
        
        return ndx * ndp > 0;
    }
    
    _HD_ bool in_tetrahedron(const float *x, const float *A, const float *B, const float *C, const float *D)
    {
        const float AB[3] = {B[X] - A[X], B[Y] - A[Y], B[Z] - A[Z]};
        const float AC[3] = {C[X] - A[X], C[Y] - A[Y], C[Z] - A[Z]};
        const float AD[3] = {D[X] - A[X], D[Y] - A[Y], D[Z] - A[Z]};
    
        const float BC[3] = {C[X] - B[X], C[Y] - B[Y], C[Z] - B[Z]};
        const float BD[3] = {D[X] - B[X], D[Y] - B[Y], D[Z] - B[Z]};

        return
            same_side(x, A, BC, BD, D) &&
            same_side(x, B, AD, AC, D) &&
            same_side(x, C, AB, BD, D) &&
            same_side(x, D, AB, AC, A);
    }

    __global__ void inside(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= n) return;

        int count = 0;

        const Particle p = pp[gid];
        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int t1 = tt[3*i + 0];
            const int t2 = tt[3*i + 1];
            const int t3 = tt[3*i + 2];

            const float a[3] = {vv[3*t1 + 0], vv[3*t1 + 1], vv[3*t1 + 2]};
            const float b[3] = {vv[3*t2 + 0], vv[3*t2 + 1], vv[3*t2 + 2]};
            const float c[3] = {vv[3*t3 + 0], vv[3*t3 + 1], vv[3*t3 + 2]};
                
            if (in_tetrahedron(p.r, a, b, c, origin)) ++count;
        }
        
        inout[gid] = count % 2;
    }
}
