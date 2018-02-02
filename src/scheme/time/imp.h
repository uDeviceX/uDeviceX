struct Time;
struct Force;

void time_ini(float ts, float te, float dt0, /**/ Time**);
void time_dt(Time*);
void time_nxt(Time*, int n, Force*);
int time_cross(Time*, float t); /* do we just cross `n*t', where `n' is
                                   integer? */
void time_fin(Time*);
