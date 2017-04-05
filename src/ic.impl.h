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

#if 0
    // hack for faster equilibration; TODO remove that!
    __global__ void k_init_v(const int n, Particle *pp)
    {
        const int pid = threadIdx.x + blockIdx.x * blockDim.x;

        if (pid >= n)
        return;

        Particle p = pp[pid];

        const float x = p.r[X], y = p.r[Y];
        
        const float vc[3] = {(float) (-gamma_dot*0.5*y), (float) (gamma_dot*0.5*x), 0.f};
        const float vs[3] = {(float) (gamma_dot * y), 0.f, 0.f};

        const float w = exp(-(x*x+y*y - rcyl*rcyl)*0.5);
        
        p.v[X] = w * vc[X] + (1.f - w) * vs[X];
        p.v[Y] = w * vc[Y] + (1.f - w) * vs[Y];
        p.v[Z] = w * vc[Z] + (1.f - w) * vs[Z];

        pp[pid] = p;
    }
#endif
    
#undef X
#undef Y
#undef Z

} /* namespace ic */
