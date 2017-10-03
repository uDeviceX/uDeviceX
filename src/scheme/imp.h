namespace scheme {
struct Fparams {float a, b, c;};

void move(float mass, Particle *pp, Force *ff, int n);
void clear_vel(Particle *pp, int n);
void force(float mass, Fparams fpar, int n, const Particle* pp, /**/ Force* ff);
void restrain(const int *cc, int n, long it, /**/ Particle *pp);
}
