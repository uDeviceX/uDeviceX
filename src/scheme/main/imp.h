namespace scheme {
struct Fparams {float a, b, c;};

void move(float mass, int n, const Force*, Particle*);
void clear_vel(int n, Particle*);
void force(float mass, Fparams, int n, const Particle*, /**/ Force*);
}
