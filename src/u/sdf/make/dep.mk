$B/coords/imp.o: $B/conf.h $S/coords/imp.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/ini.h $S/coords/type.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/io/field/h5/imp.o: $S/coords/imp.h $S/io/field/h5/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h
$B/io/field/imp.o: $B/conf.h $S/coords/imp.h $S/inc/conf.h $S/inc/type.h $S/io/field/h5/imp.h $S/io/field/imp.h $S/io/field/imp/dump.h $S/io/field/imp/scalar.h $S/io/field/xmf/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h $S/utils/os.h
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.h $S/mpi/glb.h $S/utils/error.h $S/utils/imp.h
$B/math/linal/imp.o: $S/math/linal/imp.h $S/utils/error.h
$B/math/rnd/imp.o: $S/math/rnd/imp.h $S/utils/error.h $S/utils/imp.h
$B/math/tform/imp.o: $B/conf.h $S/inc/conf.h $S/math/tform/imp.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/parser/imp.o: $S/parser/imp.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/u/sdf/main.o: $B/conf.h $S/coords/ini.h $S/coords/type.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/math/tform/dev.h $S/math/tform/type.h $S/mpi/glb.h $S/mpi/wrapper.h $S/u/sdf/dev.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h $S/wall/sdf/def.h $S/wall/sdf/dev.h $S/wall/sdf/imp.h $S/wall/sdf/imp/type.h $S/wall/sdf/tex3d/type.h $S/wall/sdf/type.h $S/wvel/type.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
$B/wall/sdf/array3d/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/wall/sdf/array3d/imp.h $S/wall/sdf/array3d/type.h
$B/wall/sdf/bounce/imp.o: $B/conf.h $S/coords/dev.h $S/coords/imp.h $S/coords/type.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/math/dev.h $S/math/tform/dev.h $S/math/tform/type.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h $S/wall/sdf/bounce/dev/main.h $S/wall/sdf/bounce/imp.h $S/wall/sdf/bounce/imp/main.h $S/wall/sdf/def.h $S/wall/sdf/dev.h $S/wall/sdf/imp.h $S/wall/sdf/tex3d/type.h $S/wall/sdf/type.h $S/wvel/dev.h $S/wvel/type.h
$B/wall/sdf/field/imp.o: $B/conf.h $S/inc/conf.h $S/io/field/imp.h $S/math/tform/imp.h $S/mpi/glb.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h $S/wall/sdf/field/imp.h $S/wall/sdf/field/imp/main.h $S/wall/sdf/field/imp/sample.h $S/wall/sdf/tform/imp.h
$B/wall/sdf/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/tform/imp.h $S/math/tform/type.h $S/mpi/glb.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h $S/wall/sdf/array3d/imp.h $S/wall/sdf/bounce/imp.h $S/wall/sdf/def.h $S/wall/sdf/field/imp.h $S/wall/sdf/imp.h $S/wall/sdf/imp/gen.h $S/wall/sdf/imp/main.h $S/wall/sdf/imp/split.h $S/wall/sdf/imp/type.h $S/wall/sdf/label/imp.h $S/wall/sdf/tex3d/imp.h $S/wall/sdf/tex3d/type.h $S/wall/sdf/tform/imp.h $S/wall/sdf/type.h
$B/wall/sdf/label/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/tform/dev.h $S/math/tform/type.h $S/mpi/glb.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/kl.h $S/utils/msg.h $S/wall/sdf/dev.h $S/wall/sdf/imp.h $S/wall/sdf/label/dev/main.h $S/wall/sdf/label/imp.h $S/wall/sdf/label/imp/main.h $S/wall/sdf/tex3d/type.h $S/wall/sdf/type.h
$B/wall/sdf/tex3d/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/wall/sdf/array3d/type.h $S/wall/sdf/tex3d/imp.h $S/wall/sdf/tex3d/imp/main.h $S/wall/sdf/tex3d/type.h
$B/wall/sdf/tform/imp.o: $S/coords/imp.h $S/math/tform/imp.h $S/utils/error.h $S/wall/sdf/tform/imp.h
$B/wvel/conf.o: $S/parser/imp.h $S/utils/error.h $S/utils/imp.h $S/wvel/imp.h
$B/wvel/imp.o: $B/conf.h $S/coords/imp.h $S/coords/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h $S/wvel/imp.h $S/wvel/imp/main.h $S/wvel/imp/type.h $S/wvel/type.h
