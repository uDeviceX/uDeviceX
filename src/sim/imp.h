struct Sim;
struct TimeLine;
struct Config;
struct TimeSeg; /* time segments: tend and wall */

void sim_ini(Config*, MPI_Comm, TimeLine*, /**/ Sim**);
void sim_gen(Sim*, const Config*, TimeLine*, TimeSeg*);
void sim_strt(Sim*, const Config*, TimeLine*, TimeSeg*);
void sim_fin(Sim*);

void time_seg_ini(Config*, /**/ TimeSeg**);
void time_seg_fin(TimeSeg*);
