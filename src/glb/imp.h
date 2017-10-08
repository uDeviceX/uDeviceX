namespace glb {
extern __constant__  float r0[3];
extern __constant__  float gd;
void sim();  /* simulation wide kernel globals */
void step(); /* time step kernel globals */
}
