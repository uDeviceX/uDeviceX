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
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    UC(foo());
    m::fin();
}
