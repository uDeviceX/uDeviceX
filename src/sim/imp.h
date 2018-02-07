struct Sim;
struct Time;
struct Config;
void sim_ini(Config*, MPI_Comm, Time*, /**/ Sim**, float *tend);
void sim_gen(Sim*, Config*, Time*, float tend);
void sim_strt(Sim*, Config*, Time*, float tend);
void sim_fin(Sim*, Time*);
