struct UdxError {
    enum Type {SUCCESS, FAILED} status;
    char msg[256];
};

#define err_fill(e, t, frmt, ...) do {                                  \
        e.status = t;                                                   \
        sprintf(e.msg, "%s: %d: " frmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

#define err_handle(e) do {                      \
        err_handle0(e, __FILE__, __LINE__);     \
    } while (0)

void err_handle0(const UdxError e, const char *file, const int line);
