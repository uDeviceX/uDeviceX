namespace io { namespace field {
void dump(MPI_Comm cart, Particle *p, int n);
void scalar(MPI_Comm cart, float *data, const char *path);
}}
