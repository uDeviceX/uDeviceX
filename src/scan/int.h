namespace scan {
struct Work {
    unsigned int  *tmp;
    unsigned char *compressed;
};

void scan(const int *input, int size, /**/ int *output, /*w*/ Work *w);

void alloc_work(int size, /**/ Work *w);
void free_work(/**/ Work *w);
}
