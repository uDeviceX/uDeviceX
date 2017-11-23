namespace scheme {
struct Fparams {float a, b, c;};

/* quantaties and sizes */
struct QQ {Particle *o, *r;};
struct NN {int o, r;};

void move(float mass, int n, const Force*, Particle*);
void clear_vel(int n, Particle*);
void force(float mass, Fparams, int n, const Particle*, /**/ Force*);
void restrain(const int *cc, NN, long it, /**/ QQ);
}
