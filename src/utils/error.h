#define signal_error() UdxError::signal(__FILE__, __LINE__)
#define signal_error_extra(fmt, ...) UdxError::signal_extra(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define report_error() UdxError::report(__FILE__, __LINE__)


/* udx check */
#define UC(f) do {                              \
    f;                                          \
    report_error();                             \
    } while (0)


namespace UdxError {
void before(int line, const char *file);
void after();

void signal(const char *file, int line); 
void signal_extra(const char *file, int line, const char *fmt, ...);

void report(const char *file, int line);
}
