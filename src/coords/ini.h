struct Coords;
struct Config;

// tag::mem[]
void coords_ini(MPI_Comm cart, int Lx, int Ly, int Lz, Coords **c); // <1>
void coords_fin(Coords *c); // <2>
// end::mem[]


void coords_ini_conf(MPI_Comm cart, const Config *cfg, Coords **c);
