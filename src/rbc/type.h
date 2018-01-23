struct AreaVolume;
struct Edg {
    float a; /* equilibrium edge lengths and triangle area */
    float A;
};
struct Shape {
    int *anti; /* every edge is visited twice, what is the id of
                  the other visit? */
    Edg *edg;
    float totArea;
};

struct RbcQuants {
    int n, nc;             /* number of particles, cells            */
    Particle *pp, *pp_hst; /* vertices particles on host and device */
    float *av;             /* area and volume on device             */

    int *ii;               /* global ids (on host) */

    int nt, nv;            /* number of triangles and vertices per mesh */
    int *adj0, *adj1;      /* adjacency lists on device                 */
    int4 *tri, *tri_hst;   /* triangles: vertex indices                 */
    AreaVolume *area_volume;

    Shape shape;
};
