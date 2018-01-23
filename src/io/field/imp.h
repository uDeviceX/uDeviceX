struct Particle;
struct Coords;
namespace io { namespace field {
void dump(const Coords *coords, MPI_Comm cart, Particle *p, int n);
void scalar(const Coords *coords, MPI_Comm cart, float *data, const char *path);
}}
