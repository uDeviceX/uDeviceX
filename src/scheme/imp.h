namespace scheme {
struct Fparams {float a, b, c;};

/* quantaties and sizes */
struct QQ {Particle *o, *r;};
struct NN {int o, r;};

void move(float mass, int n, const Force *ff, Particle *pp);
void clear_vel(int n, Particle *pp);
void force(float mass, Fparams fpar, int n, const Particle* pp, /**/ Force* ff);
void restrain(const int *cc, NN nn, long it, /**/ QQ qq);
}
