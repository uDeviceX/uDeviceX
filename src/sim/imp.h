struct Sim;
struct Time;
struct Config;
struct TimeSeg; /* time segments: tend, dumps, restarts, ... */

void sim_ini(Config*, MPI_Comm, Time*, /**/ Sim**, float *tend);
void sim_gen(Sim*, Config*, Time*, float tend);
void sim_strt(Sim*, Config*, Time*, float tend);
void sim_fin(Sim*, Time*);

void time_seg_ini(Config*, /**/ TimeSeg**);
void time_seg_fin(TimeSeg*);
