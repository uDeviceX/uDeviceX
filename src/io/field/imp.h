struct IoField;
struct Particle;
struct Coords;

void io_field_ini(MPI_Comm comm, const Coords*, IoField**);
void io_field_fin(IoField*);
void io_field_dump_pp(const Coords *coords, MPI_Comm cart, IoField *io, int n, Particle *pp);

void io_field_dump_scalar(const Coords *coords, MPI_Comm cart, float *data, const char *path);
