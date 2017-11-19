#define COMM_MC(F)                              \
    do {                                        \
        oc::mpi_check((F), __FILE__, __LINE__); \
        if (oc::status() != 0) oc::report();    \
    } while (0)

/* interface used by comm */
namespace comm { namespace oc {
void mpi_check(int code, const char*, int line);
int status();
}} /* namespace */
