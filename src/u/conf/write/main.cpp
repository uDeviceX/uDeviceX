#include <stdio.h>
#include <vector_types.h>
#include <mpi.h>

#include "utils/msg.h"
#include "mpi/glb.h"
#include "utils/error.h"
#include "conf/imp.h"

static void set(Config *c) {
    conf_set_int("group.a", 0, c);

    // test: overwrite
    conf_set_int("group.a", 3, c);

    float b[] = {1, 2, 3, 4};
    conf_set_vfloat("group.b", 4, b, c);

    int3 i3;
    i3.x = 10;
    i3.y = 20;
    i3.z = 30;
    conf_set_int3("group.myint3", i3, c);
    
    // test: subgroup in existing group
    conf_set_int("group.subgroup.a", 5, c);
    conf_set_int("a", 0, c);
}

int main(int argc, char **argv) {
    Config *cfg;
    int dims[3];
    
    m::ini(&argc, &argv);
    // eat executable and dims
    m::get_dims(&argc, &argv, dims);

    conf_ini(/**/ &cfg);
    conf_read(argc, argv, /**/ cfg);

    UC(set(cfg));
    conf_write_exe(cfg, stdout);
    
    conf_fin(/**/ cfg);    

    m::fin();
}
