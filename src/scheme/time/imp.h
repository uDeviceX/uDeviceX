struct Time;
void time_ini(float start, /**/ Time**);
void time_fin(Time*);

void time_step(Time*, float dt); /* "register" a step */
int  time_cross(Time*, float interval); /* did we just cross
                                           `n*interval', where `n' is
                                           an integer? */
float time_t (Time*);   /* current time */
float time_dt (Time*);  /* current dt */
float time_dt0(Time*);  /* previous dt */
