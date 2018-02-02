struct Time;
struct Force;

void time_ini(float dt0, /**/ Time**);
float time_dt(Time*, int n, Force*);
int time_cross(Time*, float t); /* do we just cross `n*t', where `n' is
                                   integer? */
void time_fin(Time*);
