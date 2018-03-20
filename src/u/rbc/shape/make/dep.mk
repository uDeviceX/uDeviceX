$B/algo/edg/imp.o: $S/algo/edg/imp.h $S/algo/edg/imp/main.h $S/utils/error.h
$B/conf/imp.o: $S/conf/imp.h $S/conf/imp/get.h $S/conf/imp/main.h $S/conf/imp/set.h $S/conf/imp/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/io/mesh_read/imp.o: $S/io/mesh_read/imp.h $S/io/mesh_read/imp/main.h $S/io/mesh_read/imp/off.h $S/io/mesh_read/imp/ply.h $S/io/mesh_read/imp/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/math/linal/imp.o: $S/math/linal/imp.h $S/utils/error.h
$B/math/rnd/imp.o: $S/math/rnd/imp.h $S/utils/error.h $S/utils/imp.h
$B/math/tform/imp.o: $B/conf.h $S/inc/conf.h $S/math/tform/imp.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/math/tri/imp.o: $S/math/tri/dev.h $S/math/tri/imp.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/rbc/adj/imp.o: $S/algo/edg/imp.h $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/rbc/adj/imp.h $S/rbc/adj/imp/anti.h $S/rbc/adj/imp/fin.h $S/rbc/adj/imp/ini.h $S/rbc/adj/imp/map.h $S/rbc/adj/imp/type.h $S/rbc/adj/type/common.h $S/rbc/adj/type/dev.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/rbc/shape/imp.o: $S/math/tri/imp.h $S/rbc/adj/imp.h $S/rbc/adj/type/common.h $S/rbc/shape/imp.h $S/rbc/shape/imp/main.h $S/rbc/shape/imp/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/u/rbc/shape/main.o: $B/conf.h $S/conf/imp.h $S/inc/conf.h $S/inc/dev.h $S/io/mesh_read/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/rbc/adj/imp.h $S/rbc/shape/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h $S/utils/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
