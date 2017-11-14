$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/d/common.h $S/d/api.h $S/msg.h $B/conf.h $S/d/cuda/imp.h
$B/io/bop/imp.o: $S/utils/os.h $S/inc/conf.h $S/inc/type.h $S/mpi/wrapper.h $S/io/bop/imp.h $S/inc/def.h $S/mpi/type.h $S/msg.h $S/d/api.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/io/com.o: $S/utils/os.h $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/msg.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/io/diag.o: $S/inc/conf.h $S/inc/type.h $S/mpi/wrapper.h $S/msg.h $B/conf.h $S/io/diag.h $S/mpi/glb.h
$B/io/field/h5/imp.o: $S/mpi/wrapper.h $S/io/field/h5/imp.h $S/msg.h $S/mpi/glb.h
$B/io/field/imp.o: $S/io/field/imp/scalar.h $S/utils/os.h $S/inc/conf.h $S/io/field/xmf/imp.h $S/inc/type.h $S/mpi/wrapper.h $S/io/field/imp.h $S/io/field/h5/imp.h $S/io/field/imp/dump.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/io/fields_grid.o: $S/inc/conf.h $S/inc/type.h $S/utils/cc.h $S/io/fields_grid.h $S/inc/dev.h $S/d/api.h $S/msg.h $B/conf.h $S/io/field/imp.h $S/io/fields_grid/solvent.h $S/io/fields_grid/all.h
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.h $S/mpi/glb.h
$B/io/mesh.o: $S/utils/os.h $S/inc/conf.h $S/io/mesh.h $S/inc/type.h $S/io/mesh/shift/edge.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/io/mesh/main.h $S/io/mesh/shift/center.h $S/mpi/glb.h
$B/io/off.o: $S/io/off/imp.h $S/io/off.h
$B/io/ply.o: $S/io/ply/ascii.h $S/inc/type.h $S/inc/def.h $S/io/ply.h $S/msg.h $S/io/ply/bin.h $S/io/ply/common.h
$B/io/restart.o: $S/inc/conf.h $S/inc/type.h $S/inc/def.h $S/msg.h $B/conf.h $S/io/restart.h $S/mpi/glb.h
$B/io/rig.o: $S/inc/conf.h $S/inc/type.h $B/conf.h
$B/math/linal.o: $S/math/linal.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/rbc/adj/imp.o: $S/rbc/adj/imp.h $S/rbc/adj/imp/ini.h $S/msg.h $S/rbc/adj/type.h $S/rbc/adj/imp/map.h
$B/rbc/com/imp.o: $S/rbc/com/imp/fin.h $S/inc/conf.h $S/rbc/com/imp/com.h $S/inc/type.h $S/rbc/com/imp.h $S/utils/cc.h $S/inc/def.h $S/rbc/com/imp/ini.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/utils/kl.h $S/math/dev.h $S/rbc/com/dev/main.h $S/d/ker.h
$B/rbc/force/area_volume/imp.o: $S/inc/conf.h $S/rbc/force/area_volume/imp.h $S/utils/cc.h $S/inc/def.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/rbc/force/area_volume/imp/main.h $S/utils/texo.h $S/utils/texo.dev.h $S/utils/kl.h $S/math/dev.h $S/rbc/force/area_volume/dev/main.h $S/d/ker.h
$B/rbc/force/imp.o: $S/rbc/force/imp/fin.h $S/rbc/force/params/lina.h $S/rbc/force/params/test.h $S/rbc/force/area_volume/imp.h $S/inc/conf.h $S/rbc/force/dev/common.h $S/rbc/type.h $S/inc/type.h $S/rbc/force/dev/shape.h $S/rbc/force/imp.h $S/utils/cc.h $S/inc/def.h $S/rbc/force/imp/ini.h $S/rbc/adj/dev.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/rbc/force/dev/force.h $S/utils/texo.h $S/rbc/force/imp/forces.h $S/utils/texo.dev.h $S/utils/te.h $S/utils/kl.h $S/math/dev.h $S/rbc/adj/type.h $S/rbc/force/dev/main.h $S/d/ker.h
$B/rbc/gen/imp.o: $S/inc/conf.h $S/inc/type.h $S/mpi/wrapper.h $S/rbc/gen/imp.h $S/io/off.h $S/inc/def.h $S/utils/mc.h $S/msg.h $B/conf.h $S/rbc/gen/imp/main.h $S/mpi/glb.h
$B/rbc/main/imp.o: $S/rbc/main/imp/fin.h $S/inc/conf.h $S/rbc/type.h $S/inc/type.h $S/rbc/main/imp/util.h $S/rbc/gen/imp.h $S/io/restart.h $S/mpi/wrapper.h $S/rbc/main/imp.h $S/io/off.h $S/rbc/main/imp/setup.h $S/utils/cc.h $S/inc/def.h $S/rbc/main/imp/generate.h $S/rbc/main/imp/ini.h $S/utils/mc.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/rbc/adj/imp.h $S/rbc/main/imp/start.h $S/rbc/adj/type.h $S/mpi/glb.h
$B/rbc/stretch/imp.o: $S/inc/conf.h $S/inc/type.h $S/rbc/stretch/imp/util.h $S/rbc/stretch/imp.h $S/utils/cc.h $S/inc/def.h $S/d/api.h $S/msg.h $S/inc/dev.h $B/conf.h $S/rbc/stretch/imp/main.h $S/rbc/stretch/imp/type.h $S/utils/kl.h $S/rbc/stretch/dev/main.h $S/d/ker.h
$B/u/rbc/lib/imp.o: $S/inc/conf.h $S/rbc/type.h $S/inc/type.h $S/u/rbc/lib/imp.h $S/utils/cc.h $S/inc/def.h $S/rbc/main/imp.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/u/rbc/lib/imp/main.h $S/utils/texo.h $S/utils/te.h $S/rbc/force/imp.h $S/mpi/glb.h
$B/u/rbc/main.o: $S/u/rbc/lib/imp.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/d/api.h $S/msg.h $B/conf.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h
