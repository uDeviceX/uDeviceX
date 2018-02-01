$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/parser/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/io/field/h5/imp.o: $S/utils/error.h $S/mpi/wrapper.h $S/io/field/h5/imp.h $S/mpi/glb.h $S/coords/imp.h
$B/io/field/imp.o: $S/io/field/imp/scalar.h $S/utils/os.h $S/utils/imp.h $S/inc/conf.h $S/io/field/xmf/imp.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/io/field/imp.h $S/io/field/h5/imp.h $S/io/field/imp/dump.h $S/utils/mc.h $B/conf.h $S/coords/imp.h $S/mpi/glb.h
$B/io/field/xmf/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/field/xmf/imp.h $S/mpi/glb.h
$B/math/linal/imp.o: $S/utils/error.h $S/math/linal/imp.h
$B/math/rnd/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/rnd/imp.h
$B/math/tform/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/math/tform/imp.h $B/conf.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/parser/imp.o: $S/utils/imp.h $S/utils/error.h $S/parser/imp.h $S/utils/msg.h
$B/u/sdf/main.o: $S/coords/type.h $S/inc/conf.h $S/wall/sdf/imp.h $S/utils/error.h $S/inc/type.h $S/wall/sdf/dev.h $S/mpi/wrapper.h $S/wall/sdf/imp/type.h $S/coords/ini.h $S/utils/cc.h $S/u/sdf/dev.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/math/tform/dev.h $S/math/tform/type.h $S/utils/kl.h $S/wall/sdf/type.h $S/parser/imp.h $S/wall/sdf/tex3d/type.h $S/wall/wvel/type.h $S/coords/imp.h $S/mpi/glb.h $S/utils/msg.h $S/d/ker.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/wall/sdf/array3d/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/wall/sdf/array3d/imp.h $S/utils/cc.h $S/d/api.h $B/conf.h $S/wall/sdf/array3d/type.h
$B/wall/sdf/bounce/imp.o: $S/wall/wvel/dev.h $S/coords/type.h $S/inc/conf.h $S/wall/sdf/imp.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/wall/sdf/dev.h $S/wall/sdf/bounce/imp.h $S/utils/cc.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/wall/sdf/bounce/imp/main.h $S/math/tform/dev.h $S/math/tform/type.h $S/utils/kl.h $S/math/dev.h $S/coords/dev.h $S/wall/sdf/type.h $S/wall/sdf/bounce/dev/main.h $S/wall/sdf/tex3d/type.h $S/wall/wvel/type.h $S/coords/imp.h $S/d/ker.h $S/utils/msg.h
$B/wall/sdf/field/imp.o: $S/utils/imp.h $S/inc/conf.h $S/io/field/imp.h $S/utils/error.h $S/wall/sdf/tform/imp.h $S/wall/sdf/field/imp.h $B/conf.h $S/wall/sdf/field/imp/main.h $S/wall/sdf/field/imp/sample.h $S/math/tform/imp.h $S/coords/imp.h $S/utils/msg.h $S/mpi/glb.h
$B/wall/sdf/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/wall/sdf/label/imp.h $S/inc/type.h $S/wall/sdf/tex3d/type.h $S/wall/sdf/imp.h $S/utils/cc.h $S/inc/def.h $S/wall/sdf/imp/split.h $S/wall/sdf/imp/gen.h $S/wall/sdf/tform/imp.h $S/wall/sdf/array3d/imp.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/wall/sdf/imp/main.h $S/wall/sdf/field/imp.h $S/wall/sdf/tex3d/imp.h $S/wall/sdf/imp/type.h $S/wall/sdf/type.h $S/math/tform/type.h $S/math/tform/imp.h $S/wall/sdf/bounce/imp.h $S/coords/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/wall/sdf/label/imp.o: $S/utils/imp.h $S/inc/conf.h $S/wall/sdf/imp.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/wall/sdf/dev.h $S/wall/sdf/label/imp.h $S/utils/cc.h $S/inc/def.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/wall/sdf/label/imp/main.h $S/math/tform/dev.h $S/math/tform/type.h $S/utils/kl.h $S/wall/sdf/type.h $S/wall/sdf/label/dev/main.h $S/wall/sdf/tex3d/type.h $S/d/ker.h $S/utils/msg.h $S/mpi/glb.h
$B/wall/sdf/tex3d/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/wall/sdf/tex3d/imp.h $S/utils/cc.h $S/d/api.h $B/conf.h $S/wall/sdf/tex3d/imp/main.h $S/wall/sdf/array3d/type.h $S/wall/sdf/tex3d/type.h
$B/wall/sdf/tform/imp.o: $S/utils/error.h $S/wall/sdf/tform/imp.h $S/math/tform/imp.h $S/coords/imp.h
$B/wall/wvel/conf.o: $S/utils/imp.h $S/utils/error.h $S/wall/wvel/imp.h $B/conf.h $S/parser/imp.h
$B/wall/wvel/imp.o: $S/coords/type.h $S/utils/imp.h $S/utils/error.h $S/wall/wvel/imp.h $B/conf.h $S/wall/wvel/imp/main.h $S/wall/wvel/imp/type.h $S/wall/wvel/type.h $S/coords/imp.h $S/utils/msg.h
