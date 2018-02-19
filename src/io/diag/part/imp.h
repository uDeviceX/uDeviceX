struct Particle;
struct DiagPart;

void diag_part_ini(const char *path, /**/ DiagPart**);
void diag_part_fin(DiagPart*);

void diag(MPI_Comm comm, float time, int n, const Particle *pp);
