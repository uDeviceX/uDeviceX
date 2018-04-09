struct Sim;
struct Config;

void sim_ini(const Config*, MPI_Comm, /**/ Sim**);
void sim_gen(Sim*, const Config*);
void sim_strt(Sim*, const Config*);
void sim_fin(Sim*);
