# int
``` c++
struct Quants {
    float4 *pp;
    int n;
    Logistic::KISS *rnd;
    Clist *cells;
    cudaTextureObject_t texstart;
    cudaTextureObject_t texpp;
}
```

# wall functions called by sim::
``` c++
void alloc_quants(Quants *q);
void free_quants(Quants *q);
int create(int n, Particle* pp, Quants *q);
void interactions(const Quants q, const int type, const Particle *pp, const int n, Force *ff);
```

# bounce back

      Find wall position (sdf(wall) = 0): make two steps of Newton's
      method for the equation phi(t) = 0, where phi(t) = sdf(rr(t))
      and rr(t) = [x + vx*t, y + vy*t, z + vz*t]. We are going back
      and `t' is in [-dt, 0].

      dphi = v . grad(sdf). Newton step is t_new = t_old - phi/dphi

      Give up if dsdf is small. Cap `t' to [-dt, 0].

