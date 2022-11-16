// tag::write[]
BopStatus bop_write_header(const char *name, const BopData *d); // <1>
BopStatus bop_write_values(const char *name, const BopData *d); // <2>
// end::write[]

// tag::read[]
BopStatus bop_read_header(const char *hfname, BopData *d, char *dfname); // <1>
BopStatus bop_read_values(const char *dfname, BopData *d);               // <2>
// end::read[]
