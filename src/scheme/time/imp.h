struct Time;

// tag::interface[]
void time_ini(float start, /**/ Time**);
void time_fin(Time*);

void time_next(Time*, float dt);
int  time_cross(Time*, float interval);
float time_current (Time*);
long  time_iteration(Time*);
// end::interface[]
