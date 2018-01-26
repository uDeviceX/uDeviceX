#include <stdlib.h>
#include <stdio.h>

#include "utils/imp.h"
#include "utils/error.h"

#include "type.h"
#include "imp.h"

void scheme_move_params_ini(MoveParams ** p) {
    UC(emalloc(sizeof(MoveParams), (void**) p));
}

void scheme_move_params_fin(MoveParams * p) {
    UC(efree(p));
}

void scheme_move_params_set_timestep(float dt0, MoveParams * p) {
    p->dt0 = dt0;
}

MoveParams_v scheme_move_params_get_view(const MoveParams * p) {
    MoveParams_v v;
    v.dt0 = p->dt0;
    return v;
}
