namespace scheme {
void move(float mass, Particle *pp, Force *ff, int n);
void clear_vel(Particle *pp, int n);
void force(float mass, Particle* pp, Force* ff, int n, float driving_force0);
}
