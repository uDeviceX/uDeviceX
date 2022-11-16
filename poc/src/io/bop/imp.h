struct IoBop;

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **t);
void io_bop_fin(IoBop *t);

void io_bop_parts       (MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const char *name, int id, IoBop *t);
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp, const Force *ff, const char *name, int id, IoBop *t);
void io_bop_stresses    (MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t);
void io_bop_ids         (MPI_Comm cart, long n, const int   *ii, const char *name, int id, IoBop *t);
void io_bop_colors      (MPI_Comm cart, long n, const int   *cc, const char *name, int id, IoBop *t);
