struct Sim;
struct Config;

void sim_ini(const Config*, MPI_Comm, /**/ Sim**);
void sim_gen(Sim*);
void sim_strt(Sim*);
void sim_fin(Sim*);
