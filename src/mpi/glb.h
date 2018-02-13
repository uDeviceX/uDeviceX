namespace m { /* mini MPI */
void ini(int *argc, char ***argv);
void get_dims(int *argc, char ***argv, int dims[3]);
void get_cart(MPI_Comm comm, const int dims[3], /**/ MPI_Comm *cart);
void fin();
}
