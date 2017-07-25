namespace scan {
struct Work {
    uint *tmp;
    unsigned char *compressed;
};

void scan(const int *input, int size, /**/ int *output, /*w*/ Work *w);

void alloc_work(/**/ Work *w);
void free_work(/**/ Work *w);
}
