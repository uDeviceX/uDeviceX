namespace scan {

// tag::work[]
struct Work {
    unsigned int  *tmp;
    unsigned char *compressed;
};
// end::work[]

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Work *w); // <1>

void scan_work_ini(int size, /**/ Work *w);                                   // <2>
void scan_work_fin(/**/ Work *w);                                             // <3>
// end::interface[]

} // scan
