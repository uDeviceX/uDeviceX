$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/rbc/rnd/api/imp.o: $S/inc/conf.h $S/utils/error.h $S/rbc/rnd/api/imp.h $S/rbc/rnd/api/imp/cpu.h $S/rbc/rnd/api/imp/gaussrand.h $B/conf.h $S/utils/halloc.h $S/rbc/rnd/api/type.h $S/rbc/rnd/api/imp/cuda.h
$B/rbc/rnd/imp.o: $S/utils/os.h $S/inc/conf.h $S/rbc/rnd/imp/cu.h $S/utils/error.h $S/rbc/rnd/api/imp.h $S/rbc/rnd/imp.h $S/rbc/rnd/imp/seed.h $S/utils/cc.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/rbc/rnd/imp/main.h $S/rbc/rnd/api/type.h $S/utils/halloc.h $S/rbc/rnd/type.h $S/mpi/glb.h
$B/u/rbc/rnd/main.o: $S/inc/conf.h $S/utils/cc.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/rbc/rnd/imp.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h
