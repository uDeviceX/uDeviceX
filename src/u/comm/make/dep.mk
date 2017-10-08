$B/comm/imp.o: $S/comm/imp.h $S/comm/imp/fin.h $S/comm/imp/ini.h $S/comm/imp/main.h $B/conf.h $S/d/api.h $S/frag/imp.h $S/inc/conf.h $S/mpi/basetags.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/utils/cc.h $S/utils/mc.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/frag/imp.o: $B/conf.h $S/frag/imp.h $S/inc/conf.h
$B/glb/gdot/imp.o: $B/conf.h $S/glb/gdot/imp.h $S/glb/gdot/imp/dupire/down.h $S/glb/gdot/imp/dupire/up.h $S/glb/gdot/imp/flat.h $S/inc/conf.h $S/msg.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/gdot/imp.h $S/glb/imp.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/inc/conf.h $S/mpi/glb.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/comm/main.o: $S/comm/imp.h $S/frag/imp.h $S/glb/imp.h $S/mpi/basetags.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/msg.h $S/utils/cc/common.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
