/* structure of particle                           */
/* optional "deathlist" for particles to be killed */

struct PartList {
    const Particle *pp;
    const int *deathlist;
};

