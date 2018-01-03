struct Particle;
namespace io { namespace field {
void dump(Coords coords, MPI_Comm cart, Particle *p, int n);
void scalar(Coords coords, MPI_Comm cart, float *data, const char *path);
}}
