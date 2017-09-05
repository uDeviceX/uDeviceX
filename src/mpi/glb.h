namespace m { /* mini MPI */
extern int rank, size, coords[], dims[];
int lx(); int ly(); int lz(); /* domain size */
void ini(int argc, char **argv);
void fin();
}
