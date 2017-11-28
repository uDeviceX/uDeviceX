$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/utils/halloc.h $S/d/cuda/imp.h
$B/glb/imp.o: $S/inc/conf.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/d/api.h $B/conf.h $S/glb/imp/main.h $S/glb/imp/dec.h $S/glb/get.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $S/inc/conf.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp.h $S/msg.h $B/conf.h $S/glb/wvel/imp/dupire/common.h $S/glb/wvel/imp/dupire/down.h $S/glb/wvel/imp/dupire/up.h $S/glb/wvel/imp/sin.h
$B/io/field/h5/imp.o: $S/utils/error.h $S/mpi/wrapper.h $S/io/field/h5/imp.h $S/mpi/glb.h
$B/io/field/xmf/imp.o: $S/utils/error.h $S/io/field/xmf/imp.h $S/utils/efopen.h $S/mpi/glb.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/u/h5/main.o: $S/utils/error.h $S/io/field/h5/imp.h $S/mpi/wrapper.h $S/msg.h $S/utils/halloc.h $S/io/field/xmf/imp.h $S/mpi/glb.h
$B/utils/efopen.o: $S/utils/error.h $S/utils/efopen.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/utils/mc.h $B/conf.h
