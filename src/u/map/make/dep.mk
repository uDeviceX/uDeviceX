$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/frag/imp.o: $B/conf.h $S/frag/imp.h $S/inc/conf.h
$B/hforces/imp.o: $S/cloud/dev.h $S/cloud/imp.h $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/forces/imp.h $S/forces/pack.h $S/forces/type.h $S/forces/use.h $S/frag/imp.h $S/hforces/dev/dbg.h $S/hforces/dev/main.h $S/hforces/dev/map.h $S/hforces/imp.h $S/hforces/imp/main.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/dev/common.h $S/inc/type.h $S/mpi/glb.h $S/msg.h $S/rnd/dev.h $S/rnd/imp.h $S/utils/cc.h $S/utils/kl.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/rnd/imp.o: $S/rnd/imp.h
$B/u/map/main.o: $S/cloud/imp.h $B/conf.h $S/d/api.h $S/hforces/imp.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/mpi/glb.h $S/msg.h $S/u/map/dev.h $S/utils/cc.h $S/utils/kl.h $S/utils/map/dev.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/msg.h $S/utils/cc/common.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
$B/utils/os.o: $S/utils/os.h
