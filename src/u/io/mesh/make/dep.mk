$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/parser/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/io/mesh/imp.o: $S/utils/imp.h $S/utils/os.h $S/inc/conf.h $S/io/mesh/imp/shift/center.h $S/utils/error.h $S/io/mesh/imp/shift/edge.h $S/inc/type.h $S/mpi/wrapper.h $S/io/mesh/imp.h $S/utils/mc.h $B/conf.h $S/io/mesh/imp/main.h $S/io/mesh/imp/type.h $S/io/mesh/imp/new.h $S/io/mesh/write/imp.h $S/io/off/imp.h $S/utils/msg.h $S/coords/imp.h
$B/io/mesh/write/imp.o: $S/inc/conf.h $S/mpi/wrapper.h $S/io/mesh/write/imp.h $S/utils/mc.h $B/conf.h $S/io/mesh/write/imp/main.h
$B/io/off/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/off/imp/ply.h $S/io/off/imp.h $S/io/off/imp/main.h $S/io/off/imp/type.h $S/io/off/imp/off.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/parser/imp.o: $S/utils/imp.h $S/utils/error.h $S/parser/imp.h $S/utils/msg.h
$B/u/io/mesh/main.o: $S/inc/conf.h $S/utils/error.h $S/io/mesh/imp.h $S/mpi/wrapper.h $S/coords/ini.h $S/utils/mc.h $B/conf.h $S/parser/imp.h $S/io/off/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
