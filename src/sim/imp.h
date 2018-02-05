struct Sim;
struct Time;
void sim_ini(int argc, char **argv, MPI_Comm, /**/ Sim**, Time**);
void sim_gen(Sim*, Time*);
void sim_strt(Sim*, Time*);
void sim_fin(Sim*, Time*);
