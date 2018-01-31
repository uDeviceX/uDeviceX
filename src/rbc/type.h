struct AreaVolume;
struct Particle;
struct Adj_v;

struct Edg {
    float a; /* equilibrium edge lengths and triangle area */
    float A;
};
struct Shape {
    int *anti; /* every edge is visited twice, what is the id of
                  another visit? */
    Edg *edg;
    float totArea;
};

struct RbcQuants {
    int n, nc;             /* number of particles, cells            */
    int nt, nv;            /* number of triangles and vertices per mesh */
    Particle *pp, *pp_hst; /* vertices particles on host and device */
    int *ii;               /* global ids on host */
    Adj_v *adj_v;          /* to walk over mesh on dev */
    AreaVolume *area_volume; /* to compute area and volume */
    Shape shape;
};
