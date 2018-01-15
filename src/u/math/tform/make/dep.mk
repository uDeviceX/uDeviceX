$B/comm/imp.o: $S/comm/imp.h $S/comm/imp/fin.h $S/comm/imp/ini.h $S/comm/imp/main.h $S/comm/imp/type.h $B/conf.h $S/d/api.h $S/frag/imp.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h $S/utils/msg.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/frag/imp.o: $B/conf.h $S/frag/imp.h $S/inc/conf.h
$B/glob/imp.o: $B/conf.h $S/glob/imp.h $S/glob/imp/main.h $S/glob/ini.h $S/glob/type.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/math/linal/imp.o: $S/math/linal/imp.h $S/utils/error.h
$B/math/rnd/imp.o: $S/math/rnd/imp.h $S/utils/error.h $S/utils/imp.h
$B/math/tform/imp.o: $B/conf.h $S/inc/conf.h $S/math/tform/imp.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/sdf/tform/imp.o: $S/glob/imp.h $S/glob/type.h $S/math/tform/imp.h $S/sdf/tform/imp.h $S/utils/error.h
$B/u/math/tform/lib/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/math/tform/dev.h $S/math/tform/imp.h $S/math/tform/type.h $S/u/math/tform/lib/imp.h $S/utils/cc.h $S/utils/kl.h
$B/u/math/tform/main.o: $S/glob/imp.h $S/glob/ini.h $S/glob/type.h $S/math/tform/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/sdf/tform/imp.h $S/u/math/tform/lib/imp.h $S/utils/error.h $S/utils/msg.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
