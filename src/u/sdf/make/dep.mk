$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/halloc.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/get.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/inc/conf.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $B/conf.h $S/glb/wvel/imp.h $S/glb/wvel/imp/dupire/common.h $S/glb/wvel/imp/dupire/down.h $S/glb/wvel/imp/dupire/up.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp/sin.h $S/inc/conf.h $S/msg.h
$B/io/field/h5/imp.o: $S/io/field/h5/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h
$B/io/field/imp.o: $B/conf.h $S/inc/conf.h $S/inc/type.h $S/io/field/h5/imp.h $S/io/field/imp.h $S/io/field/imp/dump.h $S/io/field/imp/scalar.h $S/io/field/xmf/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h $S/utils/os.h
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.h $S/mpi/glb.h $S/utils/efopen.h $S/utils/error.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
<<<<<<< HEAD
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/sdf/field/imp.o: $S/inc/conf.h $S/io/field/imp.h $S/utils/error.h $S/inc/type.h $S/sdf/field/imp.h $S/utils/efopen.h $S/msg.h $B/conf.h $S/utils/halloc.h $S/mpi/glb.h
$B/sdf/imp.o: $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/sdf/imp.h $S/inc/macro.h $S/utils/cc.h $S/inc/def.h $S/sdf/sub/imp.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/sdf/imp/main.h $S/utils/halloc.h $S/sdf/imp/type.h $S/sdf/type.h $S/mpi/glb.h $S/d/ker.h
$B/sdf/sub/imp.o: $S/inc/conf.h $S/inc/dev/wvel.h $S/utils/error.h $S/inc/type.h $S/d/q.h $S/sdf/field/imp.h $S/sdf/sub/imp.h $S/inc/macro.h $S/utils/cc.h $S/inc/def.h $S/sdf/type.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/sdf/sub/imp/main.h $S/utils/halloc.h $S/sdf/sub/dev/bounce.h $S/utils/kl.h $S/glb/get.h $S/sdf/sub/dev/main.h $S/sdf/sub/dev/cheap.h $S/d/ker.h $S/mpi/glb.h
$B/u/sdf/main.o: $S/inc/conf.h $S/utils/error.h $S/sdf/imp/type.h $S/inc/type.h $S/mpi/wrapper.h $S/inc/macro.h $S/utils/cc.h $S/u/sdf/dev.h $S/sdf/type.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/sdf/imp.h $S/utils/kl.h $S/sdf/sub/dev/main.h $S/mpi/glb.h $S/d/ker.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/efopen.o: $S/utils/error.h $S/utils/efopen.h
=======
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/sdf/field/imp.o: $B/conf.h $S/inc/conf.h $S/inc/type.h $S/io/field/imp.h $S/mpi/glb.h $S/msg.h $S/sdf/field/imp.h $S/utils/efopen.h $S/utils/error.h $S/utils/halloc.h
$B/sdf/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/macro.h $S/inc/type.h $S/mpi/glb.h $S/msg.h $S/sdf/imp.h $S/sdf/imp/main.h $S/sdf/imp/type.h $S/sdf/sub/imp.h $S/sdf/type.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h
$B/sdf/sub/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/glb/get.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/dev/wvel.h $S/inc/macro.h $S/inc/type.h $S/mpi/glb.h $S/msg.h $S/sdf/field/imp.h $S/sdf/sub/dev/bounce.h $S/sdf/sub/dev/cheap.h $S/sdf/sub/dev/main.h $S/sdf/sub/imp.h $S/sdf/type.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/kl.h
$B/u/sdf/main.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/dev.h $S/inc/macro.h $S/inc/type.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/sdf/imp.h $S/sdf/imp/type.h $S/sdf/sub/dev/main.h $S/sdf/type.h $S/u/sdf/dev.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/efopen.o: $S/utils/efopen.h $S/utils/error.h
>>>>>>> 4a46ac9d65e6928386522cd153c62e453c6dab11
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/error.h $S/utils/mc.h
$B/utils/os.o: $S/utils/os.h
