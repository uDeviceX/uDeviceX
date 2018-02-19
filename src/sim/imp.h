struct Sim;
struct Time;
struct Config;
struct TimeSeg; /* time segments: tend and wall */

void sim_ini(Config*, MPI_Comm, Time*, /**/ Sim**);
void sim_gen(Sim*, Config*, Time*, TimeSeg*);
void sim_strt(Sim*, Config*, Time*, TimeSeg*);
void sim_fin(Sim*);

void time_seg_ini(Config*, /**/ TimeSeg**);
void time_seg_fin(TimeSeg*);
