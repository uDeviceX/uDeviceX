$B/comm/imp.o: $S/comm/imp/fin.h $S/inc/conf.h $S/utils/error.h $S/mpi/basetags.h $S/mpi/wrapper.h $S/comm/imp.h $S/utils/cc.h $S/comm/imp/ini.h $S/d/api.h $S/utils/mc.h $S/msg.h $B/conf.h $S/comm/imp/main.h $S/utils/halloc.h $S/frag/imp.h $S/mpi/glb.h
$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/utils/halloc.h $S/d/cuda/imp.h
$B/frag/imp.o: $S/inc/conf.h $B/conf.h $S/frag/imp.h
$B/glb/imp.o: $S/inc/conf.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/d/api.h $B/conf.h $S/glb/imp/main.h $S/glb/imp/dec.h $S/glb/get.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $S/inc/conf.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp.h $S/msg.h $B/conf.h $S/glb/wvel/imp/dupire/common.h $S/glb/wvel/imp/dupire/down.h $S/glb/wvel/imp/dupire/up.h $S/glb/wvel/imp/sin.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/u/comm/main/main.o: $S/comm/imp.h $S/mpi/basetags.h $S/mpi/wrapper.h $S/msg.h $S/frag/imp.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/efopen.o: $S/utils/error.h $S/utils/efopen.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h
