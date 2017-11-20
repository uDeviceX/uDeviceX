#define signal_error()                          \
    UdxError::signal(__FILE__, __LINE__)

#define signal_error_extra(fmt, ...)                                    \
    UdxError::signal_extra(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

/* udx check */
#define UC(F) do {                                      \
        UdxError::before(__FILE__, __LINE__);           \
        F;                                              \
        if (UdxError::error())                          \
            UdxError::report(__FILE__, __LINE__);       \
        UdxError::after();                              \
    } while (0)

namespace UdxError {
void before(const char *file, int line);
void after();

void signal(const char *file, int line); 
void signal_extra(const char *file, int line, const char *fmt, ...);

bool error();
void report(const char *file, int line);
}
