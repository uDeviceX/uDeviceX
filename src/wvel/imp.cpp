#include <mpi.h>
#include <vector_types.h>

#include "glob/imp.h"
#include "type.h"

void set_params(WvelCste  p, Wvel *wv) {
    wv->type   = WALL_VEL_CSTE;
    wv->p.cste = p;
}

void set_params(WvelShear p, Wvel *wv) {
    wv->type    = WALL_VEL_SHEAR;
    wv->p.shear = p;
}
