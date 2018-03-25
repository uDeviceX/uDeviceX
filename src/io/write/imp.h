struct WriteFile;

int write_file_open(MPI_Comm, const char *path, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, const int sz, WriteFile*);
int write_master(MPI_Comm, const void *const, int sz, WriteFile*);

int write_shift_indices(MPI_Comm, int, /**/ int*);
int write_reduce(MPI_Comm, int, /**/ int*);
