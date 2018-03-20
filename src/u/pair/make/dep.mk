$B/conf/imp.o: $S/conf/imp.h $S/conf/imp/get.h $S/conf/imp/main.h $S/conf/imp/set.h $S/conf/imp/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/pair/conf.o: $S/conf/imp.h $S/pair/imp.h $S/pair/type.h $S/utils/error.h
$B/pair/imp.o: $S/pair/imp.h $S/pair/imp/main.h $S/pair/imp/type.h $S/pair/type.h $S/utils/error.h $S/utils/imp.h
$B/u/pair/main.o: $B/conf.h $S/conf/imp.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/math/dev.h $S/mpi/glb.h $S/pair/dev.h $S/pair/imp.h $S/pair/type.h $S/u/pair/imp/main.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
