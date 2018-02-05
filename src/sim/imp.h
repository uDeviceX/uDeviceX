struct Sim; 
void sim_ini(int argc, char **argv, MPI_Comm, /**/ Sim**);
void sim_gen(Sim*);
void sim_strt(Sim*);
void sim_fin(Sim*);
