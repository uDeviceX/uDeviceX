namespace ic { /* initial conditions */

#define X 0
#define Y 1
#define Z 2

int gen(Particle* pp) { /* generate particle positions and velocities */
  fprintf(stderr, "(ic::gen) IC\n");

  assert(XS * YS * ZS * numberdensity < MAX_PART_NUM);
  
  srand48(123456);
  int iz, iy, ix, l, nd = numberdensity;
  int ip = 0; /* particle index */
  float x, y, z, dr = 0.99;
  for (iz = 0; iz < ZS; iz++)
    for (iy = 0; iy < YS; iy++)
      for (ix = 0; ix < XS; ix++) {
	/* edge of a cell */
	int xlo = -0.5*XS + ix, ylo = -0.5*YS + iy, zlo = -0.5*ZS + iz;
	for (l = 0; l < nd; l++) {
	  Particle p = Particle();
	  x = xlo + dr * drand48(), y = ylo + dr * drand48(), z = zlo + dr * drand48();
	  p.r[X] = x; p.r[Y] = y; p.r[Z] = z;

#if 1
      p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
#else // just for testing purpose
      p.v[X] = y < 0 ? -1.f : 1.f;
      p.v[Y] = 0; p.v[Z] = 0;
#endif
          
	  pp[ip++] = p;
	}
      }

  int n = ip;
  fprintf(stderr, "(ic::gen) generated %d particles\n", n);
  return n;
}
    
#undef X
#undef Y
#undef Z

} /* namespace ic */
