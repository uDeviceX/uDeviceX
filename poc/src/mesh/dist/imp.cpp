#include <vector_types.h>
#include <math.h>

#include "inc/type.h"
#include "imp.h"

enum {X, Y, Z};

static float min(float a, float b) {return a < b ? a : b;}
static float max(float a, float b) {return a < b ? b : a;}
static float dot(const float x[3], const float y[3]) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; }

static void project_t(const float *a, const float *b, const float *c, const float *r, /**/ float *p) {
    const float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    const float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
    const float ar[3] = {r[0]-a[0], r[1]-a[1], r[2]-a[2]};

    float n[3] = {ab[1]*ac[2] - ab[2]*ac[1],
                  ab[2]*ac[0] - ab[0]*ac[2],
                  ab[0]*ac[1] - ab[1]*ac[0]};
    {
        const float s = 1.f / sqrt(dot(n,n));
        n[0] *= s; n[1] *= s; n[2] *= s;
    }

    const float arn = dot(ar, n);
    const float g[3] = {ar[0] - arn * n[0],
                        ar[1] - arn * n[1],
                        ar[2] - arn * n[2]};

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

    // project (u,v) onto unit triangle

    if ( (v > u - 1) && (v < u + 1) && (v > 1 - u) ) {
        const float a_ = 0.5f * (u + v - 1);
        u -= a_;
        v -= a_;
    }
    else {
        u = max(min(1.f, u), 0.f);
        v = max(min(v, 1.f-u), 0.f);
    }

    // compute projected point
    p[0] = a[0] + u * ab[0] + v * ac[0];
    p[1] = a[1] + u * ab[1] + v * ac[1];
    p[2] = a[2] + u * ab[2] + v * ac[2];
}

static float dist_from_triangle(const float *a, const float *b, const float *c, const float *r) {
    float p[3];
    project_t(a, b, c, r, /**/ p);

    const float dr[3] = {p[0] - r[0], p[1] - r[1], p[2] - r[2]};
    return sqrt(dot(dr, dr));
}

float dist_from_mesh(int nt, const int4 *tt, const float *vv, const float *r0) {
    float dmin = 1e5f;

    for (int it = 0; it < nt; ++it) {
        int4 t = tt[it];
        const int i1 = t.x;
        const int i2 = t.y;
        const int i3 = t.z;

        const float *A = vv + 3*i1;
        const float *B = vv + 3*i2;
        const float *C = vv + 3*i3;

        const float d = dist_from_triangle(A, B, C, r0);

        dmin = min(d, dmin);
    }
    return dmin;
}
