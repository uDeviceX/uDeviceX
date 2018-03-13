struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, const Particle *pp, long n, const char *name, int id, /*w*/ BopWork *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, const Particle *pp, const Force *ff, long n, const char *name, int id, /*w*/ BopWork *t);
void io_bop_stresses    (MPI_Comm cart, const float *ss, long n, const char *name, int id);
void io_bop_ids         (MPI_Comm cart, const int *ii, long n, const char *name, int id);
void io_bop_colors      (MPI_Comm cart, const int *ii, long n, const char *name, int id);
