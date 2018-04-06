struct TimeLine;

// tag::interface[]
void time_line_ini(float start, /**/ TimeLine**);
void time_line_fin(TimeLine*);

void time_line_advance(float dt, TimeLine*);

int   time_line_cross(const TimeLine*, float interval);
float time_line_get_current (const TimeLine*);
long  time_line_get_iteration(const TimeLine*);
// end::interface[]
