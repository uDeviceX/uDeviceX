struct BopWork;

void bop_ini(MPI_Comm comm, int maxp, BopWork **t);
void bop_fin(BopWork *t);

void bop_parts(MPI_Comm cart, const Coords *coords, const Particle *pp, long n, const char *name, int id, /*w*/ BopWork *t);
void bop_parts_forces(MPI_Comm cart, const Coords *coords, const Particle *pp, const Force *ff, long n, const char *name, int id, /*w*/ BopWork *t);
void bop_ids(MPI_Comm cart, const int *ii, long n, const char *name, int id);
void bop_colors(MPI_Comm cart, const int *ii, long n, const char *name, int id);
