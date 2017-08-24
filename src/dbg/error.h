namespace dbg {
namespace dev {

enum ERR_TYPE {
    NONE,     /* no error      */
    INVALID,  /* invalid value */
};

__device__ ERR_TYPE error;

namespace err {

void ini() {
    CC(MemcpyToSymbol(error, NONE, sizeof(ERR_TYPE)));
}

static void errmsg(ERR_TYPE e) {
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
    ERR_TYPE err;
    CC(MemcpyFromSymbol(&err, error, sizeof(ERR_TYPE)));
    errmsg(err);
}

} // err
} // dev
} // dbg
