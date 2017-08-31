namespace dbg {

namespace err {

#define ERRLIST(_) _(NONE),  /* no error */     \
        _(INVALID),  /* invalid value */        \
        _(INF_VAL),  /* inf value */            \
        _(NAN_VAL)   /* nan value */

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
static __device__ err_type error;
} // dev

namespace err {

static void ini() {
    err_type e = NONE;
    CC(d::MemcpyToSymbol(&dev::error, &e, sizeof(err_type)));
}

static err_type get_err() {
    err_type e;
    CC(d::MemcpyFromSymbol(&e, &dev::error, sizeof(err_type)));
    return e;
}

static void errmsg(int line, const char *file, err_type e, const char *fun, const char *msg = "") {
    if (e != NONE) {
        fprintf(stderr, "%s:%d: (%s): %s %s\n", file, line, fun, err_str[e], msg);
    }
}

static void handle(int line, const char *file, const char *fun, const char *msg) {
    err_type e = get_err();
    errmsg(line, file, e, fun, msg);
}

static const char* get_err_str(err_type e) {return err_str[e];}

} // err
} // dbg
