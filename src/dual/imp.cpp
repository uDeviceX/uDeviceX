#include <stdio.h>
#include <math.h>
#include <conf.h>

#include "inc/conf.h"
#include "msg.h"
#include "utils/cc.h"
#include "d/api.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"

namespace dual {
void alloc(I *p, int n) {
    Palloc(&p->D, n);
    Link(&p->DP, p->D);
}

void dealloc(I p) {
    int *D;
    D = p.D;
    Pfree0(D);
}
}
