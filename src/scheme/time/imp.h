struct Time;
struct Force;

void time_ini(float ts, /**/ Time**);
void time_fin(Time*);

void time_step(Time*, float dt);
int time_cross(Time*, float t); /* do we just cross `n*t', where `n'
                                   is an integer? */
float time_current(Time*);
