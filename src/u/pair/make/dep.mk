$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp/set.h $S/conf/imp.h $S/conf/imp/main.h $S/conf/imp/type.h $S/conf/imp/get.h $S/utils/msg.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/pair/conf.o: $S/utils/error.h $S/conf/imp.h $S/pair/imp.h $S/pair/type.h
$B/pair/imp.o: $S/utils/imp.h $S/utils/error.h $S/pair/imp.h $S/pair/imp/main.h $S/pair/imp/type.h $S/pair/type.h
$B/u/pair/main.o: $S/inc/conf.h $S/utils/error.h $S/conf/imp.h $S/inc/def.h $S/utils/cc.h $S/pair/type.h $S/pair/imp.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/u/pair/imp/main.h $S/pair/dev.h $S/utils/kl.h $S/math/dev.h $S/utils/msg.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/string/imp.h
