#include <stdio.h>
#include "utils/imp.h"
#include "utils/error.h"

#include "imp.h"

enum {SUCCESS, PACK_FAILURE};
struct DFluStatus {
    int errorcode;
    int cap, count;
};

void dflu_status_ini(DFluStatus **ps) {
    DFluStatus *s;
    UC(emalloc(sizeof(DFluStatus), (void**)&s));
    s->errorcode = 0;
    *ps = s;
}

void dflu_status_fin(DFluStatus *s) {
    UC(efree(s));
}

int  dflu_status_success(DFluStatus *s) {
    return 0;
}

void dflu_status_log(DFluStatus *s) {
}

void dflu_status_over(int count, int cap, /**/ DFluStatus *s) {
 
}
