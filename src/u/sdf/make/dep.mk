$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/utils/halloc.h $S/d/cuda/imp.h
$B/glb/imp.o: $S/inc/conf.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/d/api.h $B/conf.h $S/glb/imp/main.h $S/glb/imp/dec.h $S/glb/get.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $S/inc/conf.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp.h $S/msg.h $B/conf.h $S/glb/wvel/imp/dupire/common.h $S/glb/wvel/imp/dupire/down.h $S/glb/wvel/imp/dupire/up.h $S/glb/wvel/imp/sin.h
$B/io/field/h5/imp.o: $S/utils/error.h $S/mpi/wrapper.h $S/io/field/h5/imp.h $S/mpi/glb.h
$B/io/field/imp.o: $S/io/field/imp/scalar.h $S/utils/os.h $S/inc/conf.h $S/io/field/xmf/imp.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/io/field/imp.h $S/io/field/h5/imp.h $S/io/field/imp/dump.h $S/utils/mc.h $B/conf.h $S/utils/halloc.h $S/mpi/glb.h
$B/io/field/xmf/imp.o: $S/utils/error.h $S/io/field/xmf/imp.h $S/utils/efopen.h $S/mpi/glb.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/sdf/bounce/imp.o: $S/inc/conf.h $S/inc/dev/wvel.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/bounce/imp.h $S/inc/macro.h $S/utils/cc.h $S/sdf/sub/dev/cheap.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/sdf/bounce/imp/main.h $S/utils/kl.h $S/glb/get.h $S/sdf/sub/dev/main.h $S/sdf/bounce/dev/main.h $S/d/ker.h
$B/sdf/field/imp.o: $S/inc/conf.h $S/io/field/imp.h $S/utils/error.h $S/inc/type.h $S/sdf/field/imp.h $S/utils/efopen.h $S/msg.h $B/conf.h $S/utils/halloc.h $S/mpi/glb.h
$B/sdf/imp.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/sdf/imp.h $S/inc/macro.h $S/utils/cc.h $S/inc/def.h $S/sdf/sub/imp.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/sdf/imp/main.h $S/utils/halloc.h $S/sdf/imp/type.h $S/sdf/type.h $S/sdf/bounce/imp.h $S/mpi/glb.h $S/d/ker.h
$B/sdf/label/imp.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/label/imp.h $S/inc/macro.h $S/utils/cc.h $S/inc/def.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/sdf/label/imp/main.h $S/utils/halloc.h $S/utils/kl.h $S/sdf/sub/dev/main.h $S/sdf/label/dev/main.h $S/d/ker.h $S/mpi/glb.h
$B/sdf/sub/imp.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/field/imp.h $S/sdf/sub/imp.h $S/inc/macro.h $S/utils/cc.h $S/inc/def.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/sdf/sub/imp/main.h $S/utils/halloc.h $S/utils/kl.h $S/sdf/label/imp.h $S/sdf/sub/dev/main.h $S/d/ker.h $S/mpi/glb.h
$B/u/sdf/main.o: $S/inc/conf.h $S/utils/error.h $S/sdf/imp/type.h $S/inc/type.h $S/mpi/wrapper.h $S/inc/macro.h $S/utils/cc.h $S/u/sdf/dev.h $S/sdf/type.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/sdf/imp.h $S/utils/kl.h $S/sdf/sub/dev/main.h $S/mpi/glb.h $S/d/ker.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/efopen.o: $S/utils/error.h $S/utils/efopen.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/msg.h
