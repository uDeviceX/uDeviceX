$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/dbg/conf.o: $S/dbg/imp.h $S/parser/imp.h $S/utils/error.h
$B/dbg/imp.o: $B/conf.h $S/d/api.h $S/dbg/dev/clist.h $S/dbg/dev/color.h $S/dbg/dev/common.h $S/dbg/dev/force.h $S/dbg/dev/pos.h $S/dbg/dev/vel.h $S/dbg/error.h $S/dbg/imp.h $S/dbg/imp/main.h $S/dbg/imp/type.h $S/glob/imp.h $S/glob/type.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/io/punto/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/kl.h $S/utils/msg.h
$B/glob/imp.o: $B/conf.h $S/glob/imp.h $S/glob/imp/main.h $S/glob/ini.h $S/glob/type.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/io/punto/imp.o: $S/inc/type.h $S/io/punto/imp.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/parser/imp.o: $S/parser/imp.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/u/dbg/main.o: $B/conf.h $S/d/api.h $S/dbg/imp.h $S/glob/ini.h $S/glob/type.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/mpi/glb.h $S/mpi/wrapper.h $S/parser/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
