struct Outflow {
    int *kk;        /* die or stay alive?      */
    int *ndead_dev; /* number of kills on dev  */
    int ndead;      /* number of kils          */
};

void ini(int maxp, /**/ Outflow *o);
void fin(/**/ Outflow *o);

void filter_particles_circle(float R, int n, const Particle *pp, Outflow *o);
void filter_particles_plane(float3 normal, float3 r, int n, const Particle *pp, Outflow *o);
