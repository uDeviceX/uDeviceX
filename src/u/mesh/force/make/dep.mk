$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/mesh/force/kantor0/imp.o: $S/utils/error.h $S/mesh/force/kantor0/imp.h $S/mesh/force/kantor0/dev.h $S/math/dev.h $S/utils/msg.h
$B/mesh/force/kantor1/imp.o: $S/utils/error.h $S/mesh/force/kantor1/imp.h $S/mesh/force/kantor1/dev.h $S/math/dev.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/u/mesh/force/main.o: $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $S/mesh/force/kantor0/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/nvtx/imp.o: $S/utils/error.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/error.h $S/utils/string/imp.h
