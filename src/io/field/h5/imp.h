namespace h5 {
void write(MPI_Comm cart, const char *path,
           float *data[],
           const char **names,
           int n,
           int sx, int sy, int sz);
}
