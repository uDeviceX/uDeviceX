$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/glob/imp.o: $S/inc/conf.h $S/glob/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/glob/imp.h $S/utils/mc.h $B/conf.h $S/glob/imp/main.h $S/glob/type.h
$B/io/field/h5/imp.o: $S/utils/error.h $S/mpi/wrapper.h $S/io/field/h5/imp.h $S/glob/imp.h $S/glob/type.h $S/mpi/glb.h
$B/io/field/imp.o: $S/io/field/imp/scalar.h $S/utils/os.h $S/utils/imp.h $S/inc/conf.h $S/io/field/xmf/imp.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/io/field/imp.h $S/io/field/h5/imp.h $S/io/field/imp/dump.h $S/glob/imp.h $S/utils/mc.h $B/conf.h $S/glob/type.h $S/mpi/glb.h
$B/io/field/xmf/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/field/xmf/imp.h $S/mpi/glb.h
$B/math/linal/imp.o: $S/utils/error.h $S/math/linal/imp.h
$B/math/rnd/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/rnd/imp.h
$B/math/tform/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/math/tform/imp.h $B/conf.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/sdf/array3d/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/sdf/array3d/imp.h $S/utils/cc.h $S/d/api.h $B/conf.h $S/sdf/array3d/type.h
$B/sdf/bounce/imp.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/bounce/imp.h $S/wvel/type.h $S/wvel/dev.h $S/glob/dev.h $S/utils/cc.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/sdf/bounce/imp/main.h $S/math/tform/dev.h $S/sdf/def.h $S/sdf/imp.h $S/math/tform/type.h $S/glob/type.h $S/utils/kl.h $S/sdf/tex3d/type.h $S/math/dev.h $S/sdf/dev.h $S/sdf/bounce/dev/main.h $S/d/ker.h $S/utils/msg.h
$B/sdf/dev.o: $S/sdf/tex3d/type.h $S/sdf/imp.h $S/sdf/tex3d/imp.h $S/sdf/type.h $S/sdf/imp/type.h $S/math/tform/type.h $S/math/tform/imp.h
$B/sdf/field/imp.o: $S/utils/imp.h $S/inc/conf.h $S/io/field/imp.h $S/utils/error.h $S/sdf/tform/imp.h $S/sdf/field/imp.h $S/glob/imp.h $B/conf.h $S/sdf/field/imp/main.h $S/sdf/field/imp/sample.h $S/glob/type.h $S/math/tform/imp.h $S/utils/msg.h $S/mpi/glb.h
$B/sdf/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/sdf/label/imp.h $S/inc/type.h $S/sdf/tex3d/type.h $S/sdf/def.h $S/sdf/imp.h $S/utils/cc.h $S/inc/def.h $S/sdf/imp/split.h $S/sdf/imp/gen.h $S/sdf/tform/imp.h $S/sdf/array3d/imp.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/sdf/imp/main.h $S/sdf/field/imp.h $S/sdf/tex3d/imp.h $S/sdf/imp/type.h $S/math/tform/type.h $S/math/tform/imp.h $S/sdf/bounce/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/sdf/label/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/label/imp.h $S/utils/cc.h $S/inc/def.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/sdf/label/imp/main.h $S/math/tform/dev.h $S/sdf/imp.h $S/math/tform/type.h $S/utils/kl.h $S/sdf/tex3d/type.h $S/sdf/dev.h $S/sdf/label/dev/main.h $S/d/ker.h $S/utils/msg.h $S/mpi/glb.h
$B/sdf/tex3d/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/sdf/tex3d/imp.h $S/utils/cc.h $S/d/api.h $B/conf.h $S/sdf/tex3d/imp/main.h $S/sdf/tex3d/type.h $S/sdf/array3d/type.h
$B/sdf/tform/imp.o: $S/utils/error.h $S/sdf/tform/imp.h $S/glob/imp.h $S/glob/type.h $S/math/tform/imp.h
$B/u/sdf/main.o: $S/glob/ini.h $S/inc/conf.h $S/utils/error.h $S/sdf/imp/type.h $S/inc/type.h $S/mpi/wrapper.h $S/wvel/type.h $S/utils/cc.h $S/u/sdf/dev.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/math/tform/dev.h $S/sdf/def.h $S/math/tform/type.h $S/sdf/imp.h $S/glob/type.h $S/utils/kl.h $S/sdf/tex3d/type.h $S/sdf/dev.h $S/mpi/glb.h $S/utils/msg.h $S/d/ker.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/wvel/imp.o: $S/utils/error.h $S/wvel/imp.h $S/wvel/imp/ini.h $S/glob/imp.h $B/conf.h $S/wvel/imp/main.h $S/wvel/type.h $S/glob/type.h $S/utils/msg.h
