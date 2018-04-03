struct AreaVolume;
struct Particle;

// tag::quants[]
struct RbcQuants {
    int n, nc;             /* number of particles, cells                */
    int nt, nv;            /* number of triangles and vertices per mesh */
    int md;                /* maximum valence of the mesh               */
    Particle *pp, *pp_hst; /* vertices particles on host and device     */
    bool ids;              /* global ids active ?                       */
    int *ii;               /* global ids on host                        */
    AreaVolume *area_volume; /* to compute area and volume */
};
// end::quants[]

