#include <stdio.h>

#include "msg.h"
#include "mpi/glb.h"
#include "utils/error.h"

void bar() {
    signal_error_extra("bar failed");
}

void foo() {
    UC(bar());
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    UC(foo());
    m::fin();
}
