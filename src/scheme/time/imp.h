struct TimeLine;

// tag::interface[]
void time_line_ini(float start, /**/ TimeLine**);
void time_line_fin(TimeLine*);

void time_line_next(TimeLine*, float dt);
int  time_line_cross(TimeLine*, float interval);
float time_line_current (TimeLine*);
long  time_line_iteration(TimeLine*);
// end::interface[]
