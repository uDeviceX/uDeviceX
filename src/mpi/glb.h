namespace m { /* mini MPI */
extern int rank, size, coords[], dims[];

/* local to relative to domain edge ([g]lobal) */
float x2g(float); float y2g(float); float z2g(float);

void ini(int argc, char **argv);
void fin();
}
