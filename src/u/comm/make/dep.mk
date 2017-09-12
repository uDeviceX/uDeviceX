$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/glb.o: $B/conf.h $S/d/api.h $S/glb.h $S/inc/conf.h $S/mpi/glb.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/hw/main.o: $S/glb.h $S/mpi/glb.h $S/msg.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
