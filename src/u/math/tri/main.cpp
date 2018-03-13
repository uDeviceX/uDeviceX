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

void kahan_area(int argc, char **v) {
    enum {X, Y, Z};
    double a[3], b[3], c[3];
    if (argc != 3*3 + 1)
        ERR("kahan_area needs nine arguments");
    UC(a[X] = read_dbl(v[1])); v++;
    UC(a[Y] = read_dbl(v[1])); v++;
    UC(a[Z] = read_dbl(v[1])); v++;

    UC(b[X] = read_dbl(v[1])); v++;
    UC(b[Y] = read_dbl(v[1])); v++;
    UC(b[Z] = read_dbl(v[1])); v++;

    UC(c[X] = read_dbl(v[1])); v++;
    UC(c[Y] = read_dbl(v[1])); v++;
    UC(c[Z] = read_dbl(v[1])); v++;
    msg_print("%.17e", tri_hst::kahan_area(a, b, c));
}

void ac_bc_cross(int argc, char **v) {
    enum {X, Y, Z};
    double a[3], b[3], c[3], r[3];
    if (argc != 3*3 + 1)
        ERR("ac_bc_cross needs nine arguments");
    UC(a[X] = read_dbl(v[1])); v++;
    UC(a[Y] = read_dbl(v[1])); v++;
    UC(a[Z] = read_dbl(v[1])); v++;

    UC(b[X] = read_dbl(v[1])); v++;
    UC(b[Y] = read_dbl(v[1])); v++;
    UC(b[Z] = read_dbl(v[1])); v++;

    UC(c[X] = read_dbl(v[1])); v++;
    UC(c[Y] = read_dbl(v[1])); v++;
    UC(c[Z] = read_dbl(v[1])); v++;

    tri_hst::ac_bc_cross(a, b, c, /**/ r);
    msg_print("%.17e %.17e %.17e", r[X], r[Y], r[Z]);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);

    if (argc < 1) ERR("needs FUNC");
    if (same_str(argv[1], "kahan_area0"))
        UC(kahan_area0(--argc, ++argv));
    else if (same_str(argv[1], "kahan_area"))
        UC(kahan_area (--argc, ++argv));
    else if (same_str(argv[1], "ac_bc_cross"))
        UC(ac_bc_cross (--argc, ++argv));    
    else 
        ERR("unknown FUNC: %s", argv[1]);
    
    m::fin();
}
