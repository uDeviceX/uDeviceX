struct WriteFile;

int write_file_open(MPI_Comm cart, const char*, /**/ WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm cart, const void *const, const int sz, WriteFile*);
int write_master(MPI_Comm cart, const void *const, int sz, WriteFile*);

int write_shift_indices(MPI_Comm cart, int, /**/ int*);
int write_reduce(MPI_Comm cart, int, /**/ int*);


