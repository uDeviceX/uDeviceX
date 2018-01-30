#include <stdio.h>
#include "utils/imp.h"
#include "utils/error.h"
#include "utils/msg.h"
#include "frag/imp.h"

#include "imp.h"

enum {SUCCESS, PACK_FAILURE};
struct DFluStatus {
    int errorcode;
    int cap, cnt, fid; /* capacity, count, fragment */
};

void dflu_status_ini(DFluStatus **ps) {
    DFluStatus *s;
    UC(emalloc(sizeof(DFluStatus), (void**)&s));
    s->errorcode = SUCCESS;
    *ps = s;
}

void dflu_status_fin(DFluStatus *s) {
    UC(efree(s));
}

int  dflu_status_success(DFluStatus *s) {
    return s->errorcode == SUCCESS;
}

static void pack_failure(DFluStatus *s) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = s->cap; cnt = s->cnt; fid = s->fid;
    d[X] = fraghst::i2dx(fid);
    d[Y] = fraghst::i2dy(fid);
    d[Z] = fraghst::i2dz(fid);
    msg_print("exceed capacity, fragment %d = [%d %d %d]: %ld/%ld",
        fid, d[X], d[Y], d[Z], cnt, cap);
}
void dflu_status_log(DFluStatus *s) {
    int code;
    if (s == NULL) msg_print("DFluStatus: s == NULL");
    else {
        code = s->errorcode;
        if      (code == SUCCESS)      msg_print("DFluStatus: SUCCESS");
        else if (code == PACK_FAILURE) pack_failure(s);
        else ERR("unknown errorcode = %d\n", code);
    }
}

void dflu_status_exceed(int fid, int cnt, int cap, /**/ DFluStatus *s) {
    if (s == NULL) ERR("status == NULL");
    s->fid = fid; s->cnt = cnt; s->cap = cap;
    s->errorcode = PACK_FAILURE;
}

int dflu_status_nullp(DFluStatus *s) { return s == NULL; }
