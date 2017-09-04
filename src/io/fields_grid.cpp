#include <stdlib.h>
#include <math.h>

#include <conf.h>
#include "inc/conf.h"

#include "inc/type.h"

#include "d/api.h"

#include "field.h"
#include "utils/cc.h"
#include "inc/dev.h"

#include "fields_grid.h"

void fields_grid(QQ qq, NN nn, /*w*/ Particle* hst) {
    Particle *o;
    int n;
    o = qq.o;
    n = nn.o;
    cD2H(hst, o, n);
    h5::dump(hst, n);
}
