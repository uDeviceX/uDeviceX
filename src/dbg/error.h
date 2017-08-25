namespace dbg {

namespace err {

#define ERRLIST(_)                              \
    _(NONE),  /* no error */                    \
        _(INVALID) /* invalid value */

#define make_str(s) #s
#define make_enum(s) s

enum {
    ERRLIST(make_enum),
    NERRORS
};

static const char *err_str[NERRORS] = {ERRLIST(make_str)};

#undef ERRLIST
#undef make_str
#undef make_enum
} // err

typedef int err_type;

namespace dev {
__device__ err_type error;
} // dev

namespace err {

void ini() {
    err_type e = NONE;
    CC(d::MemcpyToSymbol(&dev::error, &e, sizeof(err_type)));
}

static void errmsg(err_type e, const char *fun, const char *msg) {
    if (e != NONE) {
        MSG("DBG: ERR (%s): %s %s", fun, err_str[e], msg);
    }
}

void handle(const char *fun, const char *msg) {
    dSync();
    err_type err;
    CC(d::MemcpyFromSymbol(&err, &dev::error, sizeof(err_type)));
    errmsg(err, fun, msg);
}

} // err
} // dbg
