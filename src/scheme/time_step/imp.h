struct TimeStep;
struct TimeStepAccel;
struct Force;
struct Config;

void time_step_accel_ini(/**/ TimeStepAccel**);
void time_step_accel_fin(TimeStepAccel*);
void time_step_accel_push(TimeStepAccel*, float m, int n, Force*);

void time_step_ini(Config*, /**/ TimeStep**);
void time_step_fin(TimeStep*);
float time_step_dt(MPI_Comm, TimeStepAccel*, TimeStep*);
