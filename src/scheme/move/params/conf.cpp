#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "parser/imp.h"

#include "imp.h"

void scheme_move_params_conf(const Config *c, MoveParams *par) {
    float dt0;

//    UC(conf_lookup_float(c, "scheme.dt", &dt0));
    dt0 = dt;

    scheme_move_params_set_timestep(dt0, /**/ par);
}
