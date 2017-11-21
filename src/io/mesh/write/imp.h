namespace write {
struct File;
void all(const void *const ptr, const int sz, File*);
int rootp();
void one(const void *const ptr, int sz, File*);
int shift(int n, /**/ int *shift0);
int fopen(const char*, /**/ File*);
int fclose(File*);
}
