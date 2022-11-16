/* use define instead of enum because of implementation details */
/* see str macro                                                */
#define CBUFSIZE 256
#define CERRSIZE 1024

/* structure to describe bop file */
struct BopData {
    long n;              /* number of particles (per node)     */
    int nvars;           /* number of variables per particle   */
    char vars[CBUFSIZE]; /* variable descriptions              */
    BopType type;    
    void *data;          /* particle data                      */
};

/* error handling */
extern char bop_error_msg[CERRSIZE];

enum {
    BOP_SUCCESS,
    BOP_BADALLOC,
    BOP_BADFILE,
    BOP_WRONGVAR,
    BOP_MISMATCH,
    BOP_WFORMAT,
    BOP_NULLPTR,
    BOP_WMPISIZE,
    BOP_OVERFLOW,
    _BOP_NERR
};
