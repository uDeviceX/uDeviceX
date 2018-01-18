struct ScanWork;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ ScanWork *w); // <1>

void scan_work_ini(int size, /**/ ScanWork **w);                                  // <2>
void scan_work_fin(/**/ ScanWork *w);                                             // <3>
// end::interface[]
