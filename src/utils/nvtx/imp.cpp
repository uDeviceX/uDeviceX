#include <string.h>
#include <nvToolsExt.h>

#include "utils/error.h"

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
