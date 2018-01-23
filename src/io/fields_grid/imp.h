/* quantaties */
struct QQ {Particle *o, *s, *r;};

/* sizes */
struct NN {int o, s, r;};
void fields_grid(const Coords*, MPI_Comm, QQ, NN, Particle*);
