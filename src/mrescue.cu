#include "common.h"
#include ".conf.h"
#include "mesh.h"

#include "mrescue.h"

namespace mrescue
{
    int *tags_hst, *tags_dev;
    
    static int min2(int a, int b) {return a < b ? a : b;}
    static int max2(int a, int b) {return a < b ? b : a;}

    void init(int n)
    {
        tags_hst = new int[n];
        CC(cudaMalloc(&tags_dev, n*sizeof(int)));
    }

    void close()
    {
        delete[] tags_hst;
        CC(cudaFree(tags_dev));
    }

    static void project_t(const float *a, const float *b, const float *c, const float *r, /**/ float *rp, float *n)
    {
        const float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
        const float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
        const float ar[3] = {r[0]-a[0], r[1]-a[1], r[2]-a[2]};

        n[0] = ab[1]*ac[2] - ab[2]*ac[1];
        n[1] = ab[2]*ac[0] - ab[0]*ac[2];
        n[2] = ab[0]*ac[1] - ab[1]*ac[0];

#define dot(x, y) (x[0]*y[0] + x[1]*y[1] + x[2]*y[2])
        {
            const float s = 1.f / sqrt(dot(n,n));
            n[0] *= s; n[1] *= s; n[2] *= s;
        }
        
        const float arn = dot(ar,n);
        const float g[3] = {r[0] - arn * n[0],
                            r[1] - arn * n[1],
                            r[2] - arn * n[2]};

        float u, v;
        {
            const float ga1 = dot(g, ab);
            const float ga2 = dot(g, ac);
            const float a11 = dot(ab, ab);
            const float a12 = dot(ab, ac);
            const float a22 = dot(ac, ac);

            const float fac = 1.f / (a11*a22 - a12*a12);
            
            u = (ga1 * a22 - ga2 * a12) * fac;
            v = (ga2 * a11 - ga1 * a12) * fac;
        }
#undef dot
        
        // project (u,v) onto unit triangle

        if ( (v > u - 1) && (v < u + 1) && (v > 1 - u) )
        {
            const float a = 0.5f * (u + v - 1);
            u -= a;
            v -= a;
        }
        else
        {
            u = max2(min2(1.f, u), 0.f);
            v = max2(min2(v, 1.f-u), 0.f);
        }

        // compute projected point
        
        rp[0] = a[0] + u * ab[0] + v * ac[0];
        rp[1] = a[1] + u * ab[1] + v * ac[1];
        rp[2] = a[2] + u * ab[2] + v * ac[2];
    }
    
    static void rescue_1p(const Particle *vv, const int *tt, const int nt, const int sid, const int nv,
                          const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *p)
    {
        // check around me if there is triangles and select the closest one

        float dr2b = 1e5f, rpb[3], nb[3];
        
        const int xcid = min2(max2(0, int (p->r[0] + XS/2)), XS-1);
        const int ycid = min2(max2(0, int (p->r[1] + YS/2)), YS-1);
        const int zcid = min2(max2(0, int (p->r[2] + ZS/2)), ZS-1);
        const int cid = xcid + XS * (ycid + YS * zcid);

        const int start = tcstarts[cid];
        const int count = tccounts[cid];
        
        for (int i = start; i < start + count; ++i)
        {
            const int tid = tcids[i];
            const int t1 = tt[3*tid + 0], t2 = tt[3*tid + 1], t3 = tt[3*tid + 2];

#define ldv(t) {vv[3*t].r[0], vv[3*t].r[1], vv[3*t].r[2]}
            const float a[3] = ldv(t1), b[3] = ldv(t2), c[3] = ldv(t3);

            float rp[3], n[3];
            project_t(a, b, c, p->r, /**/ rp, n);

            const float dr[3] = {p->r[0] - rp[0], p->r[1] - rp[1], p->r[2] - rp[2]};
            const float dr2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];

            if (dr2 < dr2b)
            {
                dr2b = dr2;
                rpb[0] = rp[0]; rpb[1] = rp[1]; rpb[2] = rp[2];
                nb[0] = n[0]; nb[1] = n[1]; nb[2] = n[2];
            }
        }

        // otherwise pick one randomly

        if (dr2b > 9e4f)
        {
            const int i = 0; // TODO

            const int tid = tcids[i];
            const int t1 = tt[3*tid + 0], t2 = tt[3*tid + 1], t3 = tt[3*tid + 2];
            const float a[3] = ldv(t1), b[3] = ldv(t2), c[3] = ldv(t3);
#undef ldv
            rpb[0] = (a[0] + b[0] + c[0]) * 0.333333f;
            rpb[1] = (a[1] + b[1] + c[1]) * 0.333333f;
            rpb[2] = (a[2] + b[2] + c[2]) * 0.333333f;

            {
                const float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
                const float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
            
                nb[0] = ab[1]*ac[2] - ab[2]*ac[1];
                nb[1] = ab[2]*ac[0] - ab[0]*ac[2];
                nb[2] = ab[0]*ac[1] - ab[1]*ac[0];
            
                const float s = 1.f / sqrt(nb[0]*nb[0] + nb[1]*nb[1] + nb[2]*nb[2]);
                nb[0] *= s; nb[1] *= s; nb[2] *= s;
            }
        }

        // new particle position
#define eps 1e-4
        p->r[0] = rpb[0] + eps * nb[0];
        p->r[1] = rpb[1] + eps * nb[1];
        p->r[2] = rpb[2] + eps * nb[2];
#undef eps
    }
    
    void rescue_hst(const Mesh m, const Particle *i_pp, const int ns, const int n,
                    const int *tcstarts, const int *tccounts, const int *tcids, /**/ Particle *pp)
    {
        mesh::inside_hst(pp, n, m, i_pp, ns, /**/ tags_hst);

        for (int i = 0; i < n; ++i)
        {
            const int tag = tags_hst[i];
            
            if (tag != -1)
            rescue_1p(i_pp, m.tt, m.nt, tag, m.nv, tcstarts, tccounts, tcids, /**/ pp + i);
        }
    }
}
