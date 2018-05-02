struct Sim;
struct Config;

// tag::mem[]
void sim_ini(const Config*, MPI_Comm, /**/ Sim**);
void sim_fin(Sim*);
// end::mem[]

// tag::run[]
void sim_gen(Sim*);  // <1>
void sim_strt(Sim*); // <2>
// end::run[]
