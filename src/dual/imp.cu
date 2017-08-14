#include <stdio.h>
#include <conf.h>

#include "inc/conf.h"
#include "msg.h"
#include "cc.h"
#include "d.h"
#include "inc/dev.h"

#include "dual/type.h"
#include "dual/int.h"

namespace dual {
void alloc(I *p, int n) {
    Palloc0(&p->D, n);
    Link(&p->DP, p->D);
    //    CC(d::HostGetDevicePointer(&p->DP, p->D, 0));
}

void dealloc(I p) {
    int *D;
    D = p.D;
    CC(cudaFreeHost(D));
}
}
