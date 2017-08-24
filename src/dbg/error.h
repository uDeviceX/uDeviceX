namespace dbg {

namespace err {
enum {
    NONE,     /* no error      */
    INVALID,  /* invalid value */
};
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

static void errmsg(err_type e) {
    switch (e) {
    case INVALID:
        MSG("DBG: ERR: Invalid");
        break;
    case NONE:
    default:
        break;
    };
}

void handle() {
    dSync();
    err_type err;
    CC(d::MemcpyFromSymbol(&err, &dev::error, sizeof(err_type)));
    errmsg(err);
}

} // err
} // dbg
