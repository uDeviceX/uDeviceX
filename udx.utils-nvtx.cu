// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/utils/nvtx/imp.cpp */
#include <string.h>
#include <nvToolsExt.h>

/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int ncolors = sizeof(colors)/sizeof(uint32_t);

static uint32_t str2color(const char *s) {
    int cid, t;
    for (t = 0; *s != '\0'; ++s) t += (int) *s;
    cid = t % ncolors;
    return colors[cid];
}

void nvtx_push(const char *name) {
    nvtxEventAttributes_t e;

    memset(&e, 0, sizeof(e));

    e.version       = NVTX_VERSION;
    e.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    e.colorType     = NVTX_COLOR_ARGB;
    e.color         = str2color(name);
    e.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    e.message.ascii = name;
    nvtxRangePushEx(&e);
}

void nvtx_pop() {
    nvtxRangePop();
}
/* end src/utils/nvtx/imp.cpp */

#line 1 "src/utils/os.cpp"
