$B/coords/imp.o: $B/conf.h $S/coords/imp.h $S/coords/imp/main.h $S/coords/ini.h $S/coords/type.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/io/field/h5/imp.o: $S/coords/imp.h $S/coords/type.h $S/io/field/h5/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h
$B/io/field/imp.o: $B/conf.h $S/coords/imp.h $S/coords/type.h $S/inc/conf.h $S/inc/type.h $S/io/field/h5/imp.h $S/io/field/imp.h $S/io/field/imp/dump.h $S/io/field/imp/scalar.h $S/io/field/xmf/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/imp.h $S/utils/mc.h $S/utils/os.h
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.h $S/mpi/glb.h $S/utils/error.h $S/utils/imp.h
$B/math/linal/imp.o: $S/math/linal/imp.h $S/utils/error.h
$B/math/rnd/imp.o: $S/math/rnd/imp.h $S/utils/error.h $S/utils/imp.h
$B/math/tform/imp.o: $B/conf.h $S/inc/conf.h $S/math/tform/imp.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/sdf/array3d/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/sdf/array3d/imp.h $S/sdf/array3d/type.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h
$B/sdf/bounce/imp.o: $B/conf.h $S/coords/dev.h $S/coords/type.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/math/dev.h $S/math/tform/dev.h $S/math/tform/type.h $S/sdf/bounce/dev/main.h $S/sdf/bounce/imp.h $S/sdf/bounce/imp/main.h $S/sdf/def.h $S/sdf/dev.h $S/sdf/imp.h $S/sdf/tex3d/type.h $S/sdf/type.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h $S/wvel/dev.h $S/wvel/type.h
$B/sdf/dev.o: $S/math/tform/imp.h $S/math/tform/type.h $S/sdf/imp.h $S/sdf/imp/type.h $S/sdf/tex3d/imp.h $S/sdf/tex3d/type.h $S/sdf/type.h
$B/sdf/field/imp.o: $B/conf.h $S/coords/imp.h $S/coords/type.h $S/inc/conf.h $S/io/field/imp.h $S/math/tform/imp.h $S/mpi/glb.h $S/sdf/field/imp.h $S/sdf/field/imp/main.h $S/sdf/field/imp/sample.h $S/sdf/tform/imp.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/sdf/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/tform/imp.h $S/mpi/glb.h $S/sdf/array3d/imp.h $S/sdf/bounce/imp.h $S/sdf/def.h $S/sdf/field/imp.h $S/sdf/imp.h $S/sdf/imp/gen.h $S/sdf/imp/main.h $S/sdf/imp/split.h $S/sdf/imp/type.h $S/sdf/label/imp.h $S/sdf/tex3d/imp.h $S/sdf/tform/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/sdf/label/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/tform/dev.h $S/math/tform/type.h $S/mpi/glb.h $S/sdf/dev.h $S/sdf/imp.h $S/sdf/label/dev/main.h $S/sdf/label/imp.h $S/sdf/label/imp/main.h $S/sdf/tex3d/type.h $S/sdf/type.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h $S/utils/kl.h $S/utils/msg.h
$B/sdf/tex3d/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/sdf/array3d/type.h $S/sdf/tex3d/imp.h $S/sdf/tex3d/imp/main.h $S/sdf/tex3d/type.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h
$B/sdf/tform/imp.o: $S/coords/imp.h $S/coords/type.h $S/math/tform/imp.h $S/sdf/tform/imp.h $S/utils/error.h
$B/u/sdf/main.o: $B/conf.h $S/coords/ini.h $S/coords/type.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/math/tform/dev.h $S/math/tform/type.h $S/mpi/glb.h $S/mpi/wrapper.h $S/sdf/def.h $S/sdf/dev.h $S/sdf/imp.h $S/sdf/imp/type.h $S/sdf/tex3d/type.h $S/sdf/type.h $S/u/sdf/dev.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h $S/utils/msg.h $S/wvel/type.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/utils/error.h $S/utils/msg.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/mc.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/error.h $S/utils/msg.h $S/utils/os.h
$B/wvel/imp.o: $B/conf.h $S/coords/imp.h $S/coords/type.h $S/utils/error.h $S/utils/msg.h $S/wvel/imp.h $S/wvel/imp/ini.h $S/wvel/imp/main.h $S/wvel/type.h
