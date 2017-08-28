namespace dbg {
/* check if particles are inside domain (size L) */
void check_pos(const Particle *pp, int n, const char *M);

/* check positions of particles after a [p]osition [u]pdate (e.g. bounce-back, update) */
void check_pos_pu(const Particle *pp, int n, const char *M);

/* check if forces are within bounds */
void check_ff(const    Force *ff, int n, const char *M);

/* check if velocities are sane */
void check_vv(const    Particle *pp, int n, const char *M);
} // dbg
