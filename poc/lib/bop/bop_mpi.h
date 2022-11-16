// tag::write[]
BopStatus bop_write_header(MPI_Comm comm, const char *name, const BopData *d); // <1>
BopStatus bop_write_values(MPI_Comm comm, const char *name, const BopData *d); // <2>
// end::write[]

// tag::read[]
BopStatus bop_read_header(MPI_Comm comm, const char *hfname, BopData *d, char *dfname);  // <1>
BopStatus bop_read_values(MPI_Comm comm, const char *dfname, BopData *d);                // <2>
// end::read[]
