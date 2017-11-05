namespace m { /* mini MPI */
extern int rank, size, coords[], dims[];
int lx(); int ly(); int lz(); /* domain size */

/* local to relative to domain edge ([g]lobal) */
float x2g(float); float y2g(float); float z2g(float);

/* local to relative to domain [c]enter  */
float x2c(float); float y2c(float); float z2c(float);

void ini(int argc, char **argv);
void fin();
}
