struct AreaVolume;
struct Particle;
struct Adj_v;

struct Shape {
    int *anti; /* every edge is visited twice, what is the id of
                  another visit? */
    float *a, *A; /* equilibrium edge and area (stress free)*/
    float a0, A0; /* equilibrium edge and area (no stress free) */
    float totArea;
};

struct RbcQuants {
    int n, nc;             /* number of particles, cells            */
    int nt, nv;            /* number of triangles and vertices per mesh */
    Particle *pp, *pp_hst; /* vertices particles on host and device */
    int *ii;               /* global ids on host */
    AreaVolume *area_volume; /* to compute area and volume */

    Adj_v *adj_v;          /* to walk over mesh on dev */
    Shape shape;
};
