#define ERRLIST(_) _(NONE),  /* no error */     \
        _(INVALID),  /* invalid value */        \
        _(INF_VAL),  /* inf value */            \
        _(NAN_VAL)   /* nan value */

#define make_str(s) #s
#define make_enum(s) ERR_##s

enum {
    ERRLIST(make_enum),
    NERRORS
};

static const char *err_str[NERRORS] = {ERRLIST(make_str)};

#undef ERRLIST
#undef make_str
#undef make_enum

typedef int err_type;

namespace devdbg {
static __device__ err_type error;
} // dev

static void err_ini() {
    err_type e = ERR_NONE;
    CC(d::MemcpyToSymbol(&devdbg::error, &e, sizeof(err_type)));
}

static err_type err_get() {
    err_type e;
    CC(d::MemcpyFromSymbol(&e, &devdbg::error, sizeof(err_type)));
    return e;
}

static const char* get_err_str(err_type e) {return err_str[e];}
