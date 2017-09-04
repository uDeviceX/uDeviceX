#include <stdlib.h>
#include <math.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/type.h"

#include "d/api.h"

#include "field.h"
#include "utils/cc.h"
#include "inc/dev.h"

void dump_grid(Particle* hst) {
    int nn = 10;
    cD2H(hst, hst, nn);
    h5::dump(hst, nn);
}
