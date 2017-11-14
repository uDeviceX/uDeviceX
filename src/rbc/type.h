namespace rbc {
struct Edg {
    float a; /* equilibrium edge lengths */
    float A; /* equilibrium triangle area */
};
struct Shape {
    Edg *edg;
    float totArea;
};

struct Quants {
    int n, nc;             /* number of particles, cells            */
    Particle *pp, *pp_hst; /* vertices particles on host and device */
    float *av;             /* area and volume on device             */

    int *ii;               /* global ids (on host) */

    int nt, nv;            /* number of triangles and vertices per mesh */
    int *adj0, *adj1;      /* adjacency lists on device                 */
    int4 *tri, *tri_hst;   /* triangles: vertex indices                 */

    Shape shape;
};

} /* namespace */
