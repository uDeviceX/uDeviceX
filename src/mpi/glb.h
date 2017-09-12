namespace m { /* mini MPI */
extern int rank, size, coords[], dims[];
int lx(); int ly(); int lz(); /* domain size */

/* local to global */
float x2g(float); float y2g(float); float z2g(float);

void ini(int argc, char **argv);
void fin();
}
