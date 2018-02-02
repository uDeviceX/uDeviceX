struct Time;
void time_ini(float start, /**/ Time**);
void time_fin(Time*);

void time_step(Time*, float dt);
int time_cross(Time*, float interval); /* do we just cross
                                          `n*interval', where `n' is
                                          an integer? */
float time_current(Time*);
