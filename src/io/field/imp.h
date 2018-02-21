struct Particle;
struct Coords;
namespace io { namespace field {
void dump_pp(const Coords *coords, MPI_Comm cart, Particle *p, int n);
void dump_scalar(const Coords *coords, MPI_Comm cart, float *data, const char *path);
}}
