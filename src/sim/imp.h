struct Sim;
struct Time;
void sim_ini(int argc, char **argv, MPI_Comm, /**/ Sim**, Time**, float* tend0);
void sim_gen(Sim*, Time*, float tend0);
void sim_strt(Sim*, Time*, float tend0);
void sim_fin(Sim*, Time*);
