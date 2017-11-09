namespace glb {
void sim();                          /* simulation wide kernel globals */
void step(long i, long e, float dt0); /* time step kernel globals : [c]urrent and [e]nd timestep, dt */
}
