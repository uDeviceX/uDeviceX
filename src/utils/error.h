#define signal_error_extra(fmt, ...)                                    \
    UdxError::signal_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

/* udx check */
#define UC(F) do {                                      \
        UdxError::stack_push(__FILE__, __LINE__);       \
        F;                                              \
        if (UdxError::error()) {                        \
            UdxError::report(__FILE__, __LINE__);       \
            UdxError::abort();                          \
        }                                               \
        UdxError::stack_pop();                          \
    } while (0)

namespace UdxError {
void stack_push(const char *file, int line);
void stack_pop();

void signal_error(const char *file, int line, const char *fmt, ...);

bool error();
void report(const char *file, int line);
void abort();
}
