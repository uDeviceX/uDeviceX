namespace solidbounce {

#define _DH_ __device__ __host__
    
    _DH_ void rvprev(float *r1, float *v1, float *f0, /**/ float *r0, float *v0);
    
    /*
      return true if a root h lies in [0, dt] (output h), false otherwise
      if 2 roots, smallest root h0 is returned (first collision)
    */
    _DH_ bool robust_quadratic_roots(float a, float b, float c, /**/ float *h);

    /* coordinate changes : local: origin com with basis e0, e1, e2, no velocity */

    _DH_ void r2local (const float *e0, const float *e1, const float *e2, const float *com, const float *rg, /**/ float *rl);
    _DH_ void r2global(const float *e0, const float *e1, const float *e2, const float *com, const float *rl, /**/ float *rg);

    _DH_ void v2local (const float *e0, const float *e1, const float *e2, const float *vg, /**/ float *vl);
    _DH_ void v2global(const float *e0, const float *e1, const float *e2, const float *vl, /**/ float *vg);

    _DH_ void bb_part_local(const float *fp, const float *vcm, const float *om, /*o*/ Particle *p1, /*w*/ Particle *p0);
    
    void bounce(const Force *ff, const int np, const int ns, /**/ Particle *pp, Solid *shst);

    void bounce_nohost(const Force *ff, const int np, const int ns, /**/ Particle *pp, Solid *sdev);
}
