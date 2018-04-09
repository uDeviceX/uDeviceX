struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*); // <1>
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*); // <2>

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);
