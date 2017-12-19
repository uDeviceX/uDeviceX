#include <stdlib.h>
#include <stdint.h>

#include "msg.h"
#include "rnd/imp.h"
#include "mpi/glb.h"
#include "utils/error.h"

static void shift(int *argc, char ***argv) {
    (*argc)--;
    (*argv)++;
}

static void assert_n(int c) {
    if (c > 0) return;
    ERR("not enough args");
}

void main0(int c, char **v) {
    int n;
    assert_n(c); n = atoi(v[0]);
}

int main(int argc, char **argv) {
    m::ini(&argc, &argv);
    UC(main0(argc, argv));
    m::fin();
}
