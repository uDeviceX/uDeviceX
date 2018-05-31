struct TimeLine;

// tag::mem[]
void time_line_ini(float start, /**/ TimeLine**);
void time_line_fin(TimeLine*);
// end::mem[]

// tag::int[]
void time_line_advance(float dt, TimeLine*);           // <1>
int  time_line_cross(const TimeLine*, float interval); // <2>
// end::int[]

// tag::strt[]
void time_line_strt_dump(MPI_Comm, const char *base, int id, const TimeLine*); // <1>
void time_line_strt_read(const char *base, int id, TimeLine*); // <2>
// end::strt[]

// tag::get[]
float time_line_get_current  (const TimeLine*); // <1>
long  time_line_get_iteration(const TimeLine*); // <2>
// end::get[]
