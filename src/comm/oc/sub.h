#define COMM_MC(F)                              \
    if (oc::status() == 0)                      \
        oc::mpi_check((F), __FILE__, __LINE__);

/* interface used by comm */
namespace comm { namespace oc {
void mpi_check(int code, const char*, int line);
int status();
}} /* namespace */
