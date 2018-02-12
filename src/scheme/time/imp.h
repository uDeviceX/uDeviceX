struct Time;

// tag::interface[]
void time_ini(float start, /**/ Time**);
void time_fin(Time*);

void time_next(Time*, float dt);
int  time_cross(Time*, float interval);
float time_current (Time*);
float time_dt (Time*);
float time_dt0(Time*);
// end::interface[]
