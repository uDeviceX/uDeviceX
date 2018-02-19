struct Particle;
struct DiagPart;

void diag_part_ini(const char *path, /**/ DiagPart**);
void diag_part_fin(DiagPart*);

void diag_part_apply(DiagPart*, MPI_Comm, float time, int n, const Particle*);
