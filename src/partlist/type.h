enum {
    ALIVE = 0,
    DEAD  = 1
};

/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

// tag::partlist[]
struct PartList {
    const Particle *pp;
    const int *deathlist;
};
// end::partlist[]
