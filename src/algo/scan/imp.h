struct Scan;

// tag::interface[]
void scan_apply(const int *input, int size, /**/ int *output, /*w*/ Scan *w); // <1>

void scan_ini(int size, /**/ Scan **w);                                  // <2>
void scan_fin(/**/ Scan *w);                                             // <3>
// end::interface[]
