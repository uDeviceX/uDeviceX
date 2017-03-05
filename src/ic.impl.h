namespace ic { /* initial conditions */

#define X 0
#define Y 1
#define Z 2

int gen(Particle* pp) { /* generate particle positions and velocities */
  srand48(0);
  int L[3] = {XS, YS, ZS};
  int iz, iy, ix, l, nd = numberdensity;
  int ip = 0; /* particle index */
  float x, y, z, dr = 0.99;
  for (iz = 0; iz < L[Z]; iz++)
    for (iy = 0; iy < L[Y]; iy++)
      for (ix = 0; ix < L[X]; ix++) {
	/* edge of a cell */
	int xlo = -L[X]/2 + ix, ylo = -L[Y]/2 + iy, zlo = -L[Z]/2 + iz;
	for (l = 0; l < nd; l++) {
	  Particle p = Particle();
	  x = xlo + dr * drand48(), y = ylo + dr * drand48(), z = zlo + dr * drand48();
	  p.r[X] = x; p.r[Y] = y; p.r[Z] = z;
	  p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
	  pp[ip++] = p;
	}
      }

  int n = ip;
  fprintf(stderr, "(simulation) generated %d particles\n", n);
  return n;
}

#undef X
#undef Y
#undef Z

} /* namespace ic */
