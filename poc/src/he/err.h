/* error codes */
enum {HE_OK, HE_IO, HE_MEMORY, HE_SYS, HE_USER, HE_INDEX, HE_NUM};

/* for user code */
#define ER(fmt, ...) ERR(HE_USER, (fmt), ##__VA_ARGS__)
#define ERR(code, fmt, ...)                                             \
    do {                                                                \
        he_err(code, __FILE__, __LINE__, (fmt), ##__VA_ARGS__);         \
        return code;                                                    \
    } while(0)

#define MSG(fmt, ...) he_msg(__FILE__, __LINE__, (fmt), ##__VA_ARGS__)

void he_err(int code, const char *file, int line, const char *fmt, ...);
void he_msg(const char *file, int line, const char *fmt, ...);
