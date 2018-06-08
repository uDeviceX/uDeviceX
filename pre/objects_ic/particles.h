
void particles_ini(int n, int Lx, int Ly, int Lz, Particles *p);
void particles_fin(Particles *p);

void particles_clear_forces(int n, Particles *p);

/* returns min distance radius (distance/2) */
real particles_interactions(real rc, int Lx, int Ly, int Lz, int n, Particles *p);

void particles_advance(real dt, int Lx, int Ly, int Lz, int n, Particles *p);


real particles_temperature(int n, const Particles *p);
void particles_rescale_v(real T0, real T, int n, Particles *p);
