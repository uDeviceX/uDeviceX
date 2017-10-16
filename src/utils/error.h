#define signal_error() UdxError::signal(__FILE__, __LINE__)
#define signal_error_extra(fmt, ...) UdxError::signal_extra(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define report_error() UdxError::report(__FILE__, __LINE__)

namespace UdxError {
void signal(const char *file, int line); 
void signal_extra(const char *file, int line, const char *fmt, ...);

void report(int line, const char *file);
}
