struct TimeStep;
struct Config;

void time_step_ini(Config*, /**/ TimeStep**);
void time_step_fin(TimeStep*);

void time_step_dt(TimeStep*);
