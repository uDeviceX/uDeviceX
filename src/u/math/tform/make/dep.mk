$B/comm/imp.o: $S/comm/imp.h $S/comm/imp/fin.h $S/comm/imp/ini.h $S/comm/imp/main.h $S/comm/imp/type.h $B/conf.h $S/d/api.h $S/frag/imp.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h $S/utils/msg.h
$B/coords/conf.o: $S/coords/ini.h $S/parser/imp.h $S/utils/error.h $S/utils/imp.h
$B/coords/imp.o: $B/conf.h $S/coords/imp.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/ini.h $S/coords/type.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/frag/imp.o: $S/frag/dev.h $S/frag/imp.h
$B/math/linal/imp.o: $S/math/linal/imp.h $S/utils/error.h
$B/math/rnd/imp.o: $S/math/rnd/imp.h $S/utils/error.h $S/utils/imp.h
$B/math/tform/imp.o: $B/conf.h $S/inc/conf.h $S/math/tform/imp.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/parser/imp.o: $S/parser/imp.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/u/math/tform/lib/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/math/tform/dev.h $S/math/tform/imp.h $S/math/tform/type.h $S/u/math/tform/lib/imp.h $S/utils/cc.h $S/utils/kl.h
$B/u/math/tform/main.o: $B/conf.h $S/coords/ini.h $S/inc/conf.h $S/math/tform/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/parser/imp.h $S/u/math/tform/lib/imp.h $S/u/math/tform/tok.h $S/utils/error.h $S/utils/msg.h $S/wall/sdf/tform/imp.h
$B/u/math/tform/tok.o: $S/utils/error.h $S/utils/imp.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
$B/wall/sdf/tform/imp.o: $S/coords/imp.h $S/math/tform/imp.h $S/utils/error.h $S/wall/sdf/tform/imp.h
