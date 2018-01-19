#include <stdio.h>

#include "utils/msg.h"
#include "utils/error.h"

#include "parser/imp.h"
#include "imp.h"

void dflu_status_ini_conf(const Config *cfg, DFluStatus **ps) {
    int dbg;
    UC(conf_lookup_bool(cfg, "dflu.debug", /**/ &dbg));
    msg_print("dflu.debug = %d", dbg);

    if (dbg) dflu_status_ini(ps);
    else     *ps = NULL;
}
