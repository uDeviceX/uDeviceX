struct Coords;
struct Rigid;

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id, int ns, const Rigid *ss);

