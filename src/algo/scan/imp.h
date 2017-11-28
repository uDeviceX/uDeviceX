namespace scan {

// tag::work[]
struct Work {
    unsigned int  *tmp;
    unsigned char *compressed;
};
// end::work[]

// tag::interface[]
void scan(const int *input, int size, /**/ int *output, /*w*/ Work *w); // <1>

void alloc_work(int size, /**/ Work *w);                                // <2>
void free_work(/**/ Work *w);                                           // <3>
// end::interface[]

} // scan
