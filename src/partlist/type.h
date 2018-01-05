enum {
    ALIVE = 0,
    DEAD  = 1
};

/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

// tag::type[]
struct PartList {
    const Particle *pp;   /* particles array          */
    const int *deathlist; /* what pp is dead or alive */
};
// end::type[]
