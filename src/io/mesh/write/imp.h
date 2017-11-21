namespace write {
struct File;
void all(const void *const, const int sz, File*);
int rootp();
int one(const void *const, int sz, File*);

int shift(int, /**/ int*);
int reduce(int, /**/ int*);

int fopen(const char*, /**/ File**);
int fclose(File*);
}
