$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/d/common.h $S/d/api.h $S/msg.h $B/conf.h $S/d/cuda/imp.h
$B/glb/imp.o: $S/inc/conf.h $S/glb/imp/util.h $S/glb/set.h $S/d/api.h $B/conf.h $S/glb/imp/main.h $S/glb/imp/dec.h $S/glb/vwall/imp.h $S/glb/get.h $S/mpi/glb.h
$B/glb/vwall/imp.o: $S/inc/conf.h $S/glb/vwall/imp/flat.h $S/glb/vwall/imp.h $S/msg.h $B/conf.h $S/glb/vwall/imp/dupire/common.h $S/glb/vwall/imp/dupire/down.h $S/glb/vwall/imp/dupire/up.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/u/hw/main.o: $S/msg.h $S/mpi/glb.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/mc.h $B/conf.h
