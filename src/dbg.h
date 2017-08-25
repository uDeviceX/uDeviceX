namespace dbg {
/* check if particles are inside domain (size L), and if velocities are within bounds */
void check_pp(const Particle *pp, int n, const char *M);

/* check positions of particles after a position update (e.g. bounce-back, update) */
void check_pp_pu(const Particle *pp, int n, const char *M);

/* check if forces are within bounds */
void check_ff(const    Force *ff, int n, const char *M);
} // dbg
