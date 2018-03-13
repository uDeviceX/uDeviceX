#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "mpi/glb.h"
#include "utils/imp.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "math/tri/imp.h"

double read_dbl(const char *v) {
    double x;
    if (sscanf(v, "%lf", &x) != 1)
        ERR("needs a double: %s", v);
    return x;
}

void kahan_area0(int argc, char **v) {
    double a, b, c;
    if (argc != 4)
        ERR("kahan_area0 needs three arguments");
    UC(a = read_dbl(v[1])); v++;
    UC(b = read_dbl(v[1])); v++;
    UC(c = read_dbl(v[1])); v++;
    msg_print("%.17e", tri_hst::kahan_area0(a, b, c));
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);

    if (argc < 1) ERR("needs FUNC");
    if (same_str(argv[1], "kahan_area0"))
        UC(kahan_area0(--argc, ++argv));
    else
        ERR("unknown FUNC: %s", argv[1]);
    
    m::fin();
}
