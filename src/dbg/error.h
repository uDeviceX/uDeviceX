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

void handle() {
    dSync();
    ERR_TYPE err;
    CC(MemcpyFromSymbol(&err, error, sizeof(ERR_TYPE)));
    // TODO
}

} // err
} // dev
} // dbg
