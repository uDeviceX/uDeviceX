$B/comm/imp.o: $S/comm/imp/fin.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/comm/imp.h $S/utils/cc.h $S/comm/imp/ini.h $S/d/api.h $S/utils/mc.h $B/conf.h $S/comm/imp/main.h $S/comm/imp/type.h $S/frag/imp.h $S/utils/msg.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/frag/imp.o: $S/frag/dev.h $S/frag/imp.h
$B/math/tri/imp.o: $S/math/tri/imp.h $S/math/tri/dev.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/u/math/tri/main.o: $S/utils/error.h $S/math/tri/imp.h $S/utils/msg.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
