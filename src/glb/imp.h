namespace glb {
extern __constant__  float r0[3];
extern __constant__  float gd;
void sim();            /* simulation wide kernel globals */
void step(long i, long e); /* time step kernel globals : current and
                              end timestep */
}
