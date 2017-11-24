#define UERR(fmt, ...)  do {                                            \
        UdxError::signal_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        UdxError::report();                                             \
        UdxError::abort();                                              \
    } while (0)

/* udx check */
#define UC(F) do {                                      \
        UdxError::stack_push(__FILE__, __LINE__);       \
        F;                                              \
        if (UdxError::error()) {                        \
            UdxError::report();                         \
            UdxError::abort();                          \
        }                                               \
        UdxError::stack_pop();                          \
    } while (0)

namespace UdxError {
void stack_push(const char *file, int line);
void stack_pop();

void signal_error(const char *file, int line, const char *fmt, ...);
void signal_cuda_error(const char *file, int line, const char *msg);
void signal_mpi_error(const char *file, int line, const char *msg);

bool error();
void report();
void abort();
}
