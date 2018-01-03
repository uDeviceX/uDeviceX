namespace write {
struct File;
void all(MPI_Comm cart, const void *const, const int sz, File*);
int one(MPI_Comm cart, const void *const, int sz, File*);

int shift(MPI_Comm cart, int, /**/ int*);
int reduce(MPI_Comm cart, int, /**/ int*);

int fopen(MPI_Comm cart, const char*, /**/ File**);
int fclose(File*);
}
