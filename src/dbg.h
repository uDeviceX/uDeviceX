namespace dbg {
/* check if particles are inside domain, and if velocities are within bounds */
void check_pp(const Particle *pp, int n, const char *M);

/* check if forces are within bounds */
void check_ff(const    Force *ff, int n, const char *M);
} // dbg
