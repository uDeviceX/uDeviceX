$B/algo/kahan_sum/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/kahan_sum/imp.h $S/algo/kahan_sum/imp/main.h $S/algo/kahan_sum/imp/type.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp.h $S/utils/msg.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/io/mesh_read/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/mesh_read/imp/ply.h $S/io/mesh_read/imp.h $S/io/mesh_read/imp/main.h $S/io/mesh_read/imp/type.h $S/io/mesh_read/imp/off.h $S/utils/msg.h
$B/math/tri/imp.o: $S/math/tri/imp.h $S/math/tri/dev.h
$B/mesh/angle/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/tri/imp.h $S/mesh/angle/imp.h $S/algo/kahan_sum/imp.h $S/io/mesh_read/imp.h $S/mesh/angle/imp/main.h $S/mesh/angle/imp/type.h $S/mesh/positions/imp.h $S/utils/msg.h
$B/mesh/positions/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/mesh/positions/imp.h $S/mesh/positions/imp/main.h $S/mesh/positions/imp/type.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/u/mesh/angle/main.o: $S/utils/error.h $S/conf/imp.h $S/mpi/wrapper.h $S/io/mesh_read/imp.h $S/utils/mc.h $S/mesh/angle/imp.h $S/mesh/positions/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
