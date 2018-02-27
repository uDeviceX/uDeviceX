$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/conf/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/dbg/conf.o: $S/utils/error.h $S/dbg/imp.h $S/conf/imp.h
$B/dbg/imp.o: $S/dbg/dev/clist.h $S/coords/type.h $S/utils/imp.h $S/inc/conf.h $S/dbg/dev/common.h $S/dbg/error.h $S/utils/error.h $S/dbg/dev/vel.h $S/inc/type.h $S/dbg/imp.h $S/utils/cc.h $S/inc/def.h $S/dbg/dev/color.h $S/io/txt/imp.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/dbg/imp/main.h $S/dbg/dev/force.h $S/dbg/dev/pos.h $S/dbg/imp/type.h $S/utils/kl.h $S/coords/imp.h $S/utils/msg.h
$B/io/txt/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/io/txt/imp.h $S/io/txt/imp/dump.h $S/io/txt/imp/read.h $S/io/txt/imp/type.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp.h $S/utils/msg.h
$B/u/dbg/main.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/dbg/imp.h $S/mpi/wrapper.h $S/coords/ini.h $S/utils/cc.h $S/inc/dev.h $S/utils/mc.h $S/d/api.h $B/conf.h $S/utils/kl.h $S/conf/imp.h $S/coords/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
