$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/glb/gdot/imp.o: $B/conf.h $S/glb/gdot/imp.h $S/glb/gdot/imp/dupire/common.h $S/glb/gdot/imp/dupire/down.h $S/glb/gdot/imp/dupire/up.h $S/glb/gdot/imp/flat.h $S/inc/conf.h $S/msg.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/gdot/imp.h $S/glb/get.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/glb/set.h $S/inc/conf.h $S/mpi/glb.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/pair/main.o: $B/conf.h $S/d/api.h $S/forces/imp.h $S/forces/type.h $S/forces/use.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/mpi/glb.h $S/u/pair/imp/main.h $S/utils/cc.h $S/utils/kl.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/msg.h $S/utils/cc/common.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
$B/utils/os.o: $S/utils/os.h
