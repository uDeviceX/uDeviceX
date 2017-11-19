#define OC(F)                                   \
    do {                                        \
        comm::oc::before(__FILE__, __LINE__);   \
        if (comm::oc::error(F))                 \
            comm::oc::report();                 \
    } while (0)

namespace comm { namespace oc {
void before(const char*, int);
int  error(int);
void report();
} } /* namespace */
