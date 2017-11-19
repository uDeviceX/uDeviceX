#define OC(F)                                   \
    do {                                        \
        comm::oc::before(__FILE__, __LINE__);   \
        if ((F) != 0) comm::oc:error();         \
        comm::oc::after();                      \
    } while (0)

namespace comm { namespace oc {
} } /* namespace */
