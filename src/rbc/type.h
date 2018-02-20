struct AreaVolume;
struct Particle;

struct Shape {
    int *anti; /* every edge is visited twice, what is the id of
                  another visit? */
};

struct RbcQuants {
    int n, nc;             /* number of particles, cells            */
    int nt, nv;            /* number of triangles and vertices per mesh */
    Particle *pp, *pp_hst; /* vertices particles on host and device */
    int *ii;               /* global ids on host */
    AreaVolume *area_volume; /* to compute area and volume */

    Shape shape;
};
