struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

// tag::amem[]
void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);
// end::amem[]

// tag::aint[]
void time_step_accel_push(TimeStepAccel*, float m, int n, Force*); // <1>
void time_step_accel_reset(TimeStepAccel*); // <2>
// end::aint[]


// tag::mem[]
void time_step_ini(const Config*, /**/ TimeStep**); // <1>
void time_step_fin(TimeStep*);
// end::mem[]

// tag::int[]
float time_step_dt(TimeStep*, MPI_Comm, TimeStepAccel*); // <1>
float time_step_dt0(TimeStep*);                          // <2>
void time_step_log(TimeStep*);                           // <3>
// end::int[]
