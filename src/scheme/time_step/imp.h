struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

// tag::accel[]
void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);
void time_step_accel_push(TimeStepAccel*, float m, int n, Force*);
// end::accel[]

// tag::interface[]
void time_step_ini(Config*, /**/ TimeStep**);
void time_step_fin(TimeStep*);
float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
float time_step_dt0(TimeStep*);
void time_step_log(TimeStep*);
// end::interface[]
