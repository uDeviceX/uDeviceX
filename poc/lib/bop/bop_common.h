struct Cbuf {
    enum {SIZ=256};
    char c[SIZ];
};
struct BopData;
typedef int BopStatus;

// tag::type[]
enum {BopFLOAT, BopDOUBLE, BopINT, BopFASCII, BopIASCII, _BopNTYPES};
typedef int BopType;
// end::type[]

// tag::mem[]
BopStatus bop_ini(BopData **d);  // <1>
BopStatus bop_alloc(BopData *d); // <2>
BopStatus bop_fin(BopData *d);   // <3>
// end::mem[]

// tag::set[]
BopStatus bop_set_n(long n, BopData *d);                     // <1>
BopStatus bop_set_vars(int n, const char *vars, BopData *d); // <2>
BopStatus bop_set_type(BopType type, BopData *d);            // <3>
void* bop_get_data(BopData *d);                              // <4>
// end::set[]

// tag::get[]
BopStatus bop_get_n(const BopData *d, long *n);              // <1>
BopStatus bop_get_nvars(const BopData *d, int *n);           // <2>
BopStatus bop_get_vars(const BopData *d, Cbuf *vars);        // <3>
BopStatus bop_get_vars(const BopData *d, const char **vars); // <4>
BopStatus bop_get_type(const BopData *d, BopType *type);     // <5>
const void* bop_get_data(const BopData *d);                  // <6>
// end::get[]

// tag::tools[]
BopStatus bop_summary(const BopData *d);                                    // <1>
BopStatus bop_concatenate(const int nd, const BopData **dd, BopData *dall); // <2>
// end::tools[]

// tag::err[]
bool         bop_success(BopStatus status);           // <1>
const char * bop_report_error_desc(BopStatus status); // <2>
char *       bop_report_error_mesg();                 // <3>
// end::err[]
