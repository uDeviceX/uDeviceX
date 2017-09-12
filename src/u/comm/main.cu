#include <stdio.h>

#include "msg.h"
#include "mpi/glb.h"
#include "mpi/basetags.h"
#include "glb.h"

#include "comm/imp.h"

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Comm unit test!");

    basetags::TagGen tg;
    Bags sendB, recvB;
    Stamp stamp;

    ini(/**/ &tg);
    ini_no_bulk(sizeof(int), 1, /**/ &sendB);
    ini_no_bulk(sizeof(int), 1, /**/ &recvB);
    ini(m::cart, /*io*/ &tg, /**/ &stamp);
    
    
    fin(&sendB);
    fin(&recvB);
    fin(/**/ &stamp);
    
    m::fin();
}
