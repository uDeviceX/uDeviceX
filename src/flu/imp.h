int gen0(Particle *pp) { /* generate particle positions and velocities */
    enum {X, Y, Z};
    assert(XS * YS * ZS * numberdensity < MAX_PART_NUM);
  
    srand48(123456);
    int iz, iy, ix, l, nd = numberdensity;
    int n = 0; /* particle index */
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

            p.v[X] = 0; p.v[Y] = 0; p.v[Z] = 0;
          
            pp[n++] = p;
        }
    }

    MSG("ic::gen: created %06d solvent particles", n);
    return n;
}

int gen(Particle *dev, /*w*/ Particle *hst) {
  int n = gen0(hst);
  cH2D(dev, hst, n);
  return n;
}
