struct Coords;
struct Solid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Solid *ss);

