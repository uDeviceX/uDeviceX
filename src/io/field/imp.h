struct Particle;
struct Coords;

void io_field_dump_pp(const Coords *coords, MPI_Comm cart, Particle *p, int n);
void io_field_dump_scalar(const Coords *coords, MPI_Comm cart, float *data, const char *path);
