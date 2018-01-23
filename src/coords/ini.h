struct Coords;

// tag::mem[]
void coords_ini(MPI_Comm cart, Coords **c); // <1>
void coords_fin(Coords *c);                 // <2>
// end::mem[]
