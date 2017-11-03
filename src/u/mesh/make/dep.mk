$B/algo/minmax.o: $S/inc/conf.h $S/inc/type.h $S/d/q.h $S/utils/cc.h $S/inc/def.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/utils/kl.h $S/algo/minmax.h $S/d/ker.h
$B/algo/scan/imp.o: $S/algo/scan/int.h $S/inc/conf.h $S/d/q.h $S/utils/cc.h $S/inc/def.h $S/algo/scan/dev.h $S/algo/scan/cpu/imp.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/utils/kl.h $S/algo/scan/cuda/imp.h $S/d/ker.h
$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/d/common.h $S/d/api.h $S/msg.h $B/conf.h $S/d/cuda/imp.h
$B/glb/gdot/imp.o: $S/inc/conf.h $S/glb/gdot/imp/flat.h $S/glb/gdot/imp.h $S/msg.h $B/conf.h $S/glb/gdot/imp/dupire/common.h $S/glb/gdot/imp/dupire/down.h $S/glb/gdot/imp/dupire/up.h
$B/glb/imp.o: $S/inc/conf.h $S/glb/imp/util.h $S/glb/set.h $S/glb/gdot/imp.h $S/d/api.h $B/conf.h $S/glb/imp/main.h $S/glb/imp/dec.h $S/glb/get.h $S/mpi/glb.h
$B/io/off.o: $S/io/off/imp.h $S/io/off.h
$B/mesh/bbox.o: $S/inc/type.h $S/algo/minmax.h $S/mesh/bbox.h
$B/mesh/collision.o: $S/inc/conf.h $S/inc/type.h $S/d/q.h $S/utils/cc.h $S/inc/def.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/utils/texo.h $S/utils/kl.h $S/utils/texo.dev.h $S/mesh/collision.h $S/d/ker.h
$B/mesh/dist.o: $S/inc/type.h $S/mesh/dist.h
$B/mesh/props.o: $S/mesh/props.h $S/inc/type.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/u/mesh/main.o: $S/inc/conf.h $S/inc/type.h $S/io/off.h $S/inc/def.h $S/utils/cc.h $S/msg.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/u/mesh/imp/main.h $S/utils/texo.h $S/utils/kl.h $S/mesh/collision.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/d/api.h $S/msg.h $B/conf.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h
