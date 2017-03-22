#include <cstdio>
#include "quadroots.h"

const float eps = 1e-8;
const float rcyl_bb = 1.;
const float dt = 1e-3;

enum {X, Y, Z};

bool intersect_dr(float *r0, float *r1, /**/ float *t)
{
    float r0x = r0[X], r0y = r0[Y];
    float r1x = r1[X], r1y = r1[Y];
    
    const float inv_r = 1.f / rcyl_bb;
    
    r0x *= inv_r; r0y *= inv_r;
    r1x *= inv_r; r1y *= inv_r;

    const float drx = r1x - r0x;
    const float dry = r1y - r0y;
    const float a = drx * drx + dry * dry;
    
    // if (a < eps)
    // return false;
    
    const float b =
        2 * r0x * (r1x - r0x) +
        2 * r0y * (r1y - r0y);
    
    const float c = r0x * r0x + r0y * r0y - 1.f;

    RealComp t0, t1;

    printf("a = %.6e, b = %.6e, c = %.6e\n", a, b, c);
    
    if (!robust_quadratic_roots(a, b, c, &t0, &t1))
    return false;

    printf("t0, t1 = (%f %f)\n", t0, t1);
    
    if (t0 > 0 && t0 < 1) {*t = t0; return true;}
    if (t1 > 0 && t1 < 1) {*t = t1; return true;}
    
    return false;
}

bool intersect_v(float *r0, float *v0, /**/ float *h)
{
    float rx = r0[X], ry = r0[Y];
    float vx = v0[X], vy = v0[Y];
    
    const float a = vx * vx + vy * vy;
    
    // if (a < eps)
    // return false;
    
    const float b = 2 * (rx * vx + ry * vy);
    const float c = rx * rx + ry * ry - rcyl_bb*rcyl_bb;

    RealComp h0, h1;

    printf("a = %.6e, b = %.6e, c = %.6e\n", a, b, c);
    
    if (!robust_quadratic_roots(a, b, c, &h0, &h1))
    return false;

    printf("h0, h1 = (%f %f)\n", h0, h1);
    
    if (h0 > 0 && h0 < dt) {*h = h0; return true;}
    if (h1 > 0 && h1 < dt) {*h = h1; return true;}
    
    return false;
}

int main()
{
    float small1 = 1e-6f, small2 = 1e-5f, theta1 = 1.0f, theta2 = 1.001f;

    float r0[3] = {(rcyl_bb + small1) * (float) cos(theta1), (rcyl_bb + small1) * (float) sin(theta1), 0.f};
    float r1[3] = {(rcyl_bb - small2) * (float) cos(theta2), (rcyl_bb - small2) * (float) sin(theta2), small1};

    float t;
    
    if (intersect_dr(r0, r1, /**/ &t))
    {
        printf("intersect dr : h = %f\n", t*dt);
    }

    float v0[3] = {(r1[X]-r0[X]) / dt,
                   (r1[Y]-r0[Y]) / dt,
                   (r1[Z]-r0[Z]) / dt};

    printf("\nv = (%f %f %f)\n\n", v0[X], v0[Y], v0[Z]);
    
    if (intersect_v(r0, v0, /**/ &t))
    {
        printf("intersect v : h = %f\n", t);
    }
    
    return 0;
}
