#ifndef dt
#include "../.conf.h"
#endif
#include "../common.h"

namespace solidbounce {
    
    void rvprev(float *r1, float *v1, float *f0, /**/ float *r0, float *v0);
    
    /*
      return true if a root h lies in [0, dt] (output h), false otherwise
      if 2 roots, smallest root h0 is returned (first collision)
    */
    bool robust_quadratic_roots(float a, float b, float c, /**/ float *h);

    /* coordinate changes : local: origin com with basis e0, e1, e2, no velocity */

    void r2local(float *e0, float *e1, float *e2, float *com, float *rg, /**/ float *rl);
    void r2global(float *e0, float *e1, float *e2, float *com, float *rl, /**/ float *rg);

    void v2local(float *e0, float *e1, float *e2, float *vg, /**/ float *vl);
    void v2global(float *e0, float *e1, float *e2, float *vl, /**/ float *vg);

    
    void bounce_part(float *fp, float *cm, float *vcm, float *om, /*o*/ Particle *p1, /*w*/ Particle *p0);

    void bounce(Force *ff, int np, float *cm, float *vcm, float *om, /**/ Particle *pp, float *r_fo, float *r_to);
}
