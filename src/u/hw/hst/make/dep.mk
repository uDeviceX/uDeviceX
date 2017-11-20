$B/comm/imp.o: $S/comm/imp.h $S/comm/imp/fin.h $S/comm/imp/ini.h $S/comm/imp/main.h $S/comm/oc/imp.h $S/comm/oc/sub.h $B/conf.h $S/d/api.h $S/frag/imp.h $S/inc/conf.h $S/mpi/basetags.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h
$B/comm/oc/imp.o: $S/comm/oc/imp.h $S/comm/oc/imp/main.h $S/comm/oc/sub.h $S/msg.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/frag/imp.o: $B/conf.h $S/frag/imp.h $S/inc/conf.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/hw/hst/main.o: $S/mpi/glb.h $S/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/msg.h $S/utils/cc/common.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
$B/utils/os.o: $S/utils/os.h
