$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/glb/gdot/imp.o: $B/conf.h $S/glb/gdot/imp.h $S/glb/gdot/imp/dupire/common.h $S/glb/gdot/imp/dupire/down.h $S/glb/gdot/imp/dupire/up.h $S/glb/gdot/imp/flat.h $S/inc/conf.h $S/msg.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/gdot/imp.h $S/glb/imp.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/inc/conf.h $S/mpi/glb.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/pair/main.o: $S/glb/imp.h $S/mpi/glb.h $S/msg.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
