$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/msg.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/get.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/inc/conf.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $B/conf.h $S/glb/wvel/imp.h $S/glb/wvel/imp/dupire/common.h $S/glb/wvel/imp/dupire/down.h $S/glb/wvel/imp/dupire/up.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp/sin.h $S/inc/conf.h $S/msg.h
$B/io/bop/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/type.h $S/io/bop/imp.h $S/mpi/glb.h $S/mpi/type.h $S/mpi/wrapper.h $S/msg.h $S/utils/mc.h $S/utils/os.h
$B/io/com.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h $S/utils/os.h
$B/io/diag.o: $B/conf.h $S/inc/conf.h $S/inc/type.h $S/io/diag.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h
$B/io/field/h5/imp.o: $S/io/field/h5/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h
$B/io/field/imp.o: $B/conf.h $S/inc/conf.h $S/inc/type.h $S/io/field/h5/imp.h $S/io/field/imp.h $S/io/field/imp/dump.h $S/io/field/imp/scalar.h $S/io/field/xmf/imp.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h $S/utils/os.h
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.h $S/mpi/glb.h
$B/io/fields_grid.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/io/field/imp.h $S/io/fields_grid.h $S/io/fields_grid/all.h $S/io/fields_grid/solvent.h $S/msg.h $S/utils/cc.h
$B/io/mesh/imp.o: $B/conf.h $S/inc/conf.h $S/inc/type.h $S/io/mesh/imp.h $S/io/mesh/imp/main.h $S/io/mesh/imp/shift/center.h $S/io/mesh/imp/shift/edge.h $S/io/mesh/write/imp.h $S/mpi/glb.h $S/utils/error.h $S/utils/halloc.h $S/utils/os.h
$B/io/mesh/write/imp.o: $B/conf.h $S/inc/conf.h $S/io/mesh/write/imp.h $S/io/mesh/write/imp/main.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/io/off.o: $S/io/off.h $S/io/off/imp.h
$B/io/ply.o: $S/inc/def.h $S/inc/type.h $S/io/ply.h $S/io/ply/ascii.h $S/io/ply/bin.h $S/io/ply/common.h $S/msg.h
$B/io/restart.o: $B/conf.h $S/inc/conf.h $S/inc/def.h $S/inc/type.h $S/io/restart.h $S/mpi/glb.h $S/msg.h
$B/io/rig.o: $B/conf.h $S/inc/conf.h $S/inc/type.h
$B/math/linal.o: $S/math/linal.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/rbc/adj/imp.o: $S/msg.h $S/rbc/adj/imp.h $S/rbc/adj/imp/fin.h $S/rbc/adj/imp/ini.h $S/rbc/adj/imp/map.h $S/rbc/adj/type/common.h $S/rbc/adj/type/hst.h $S/rbc/edg/imp.h $S/utils/error.h $S/utils/halloc.h
$B/rbc/com/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/dev.h $S/msg.h $S/rbc/com/dev/main.h $S/rbc/com/imp.h $S/rbc/com/imp/com.h $S/rbc/com/imp/fin.h $S/rbc/com/imp/ini.h $S/utils/cc.h $S/utils/kl.h
$B/rbc/edg/imp.o: $S/msg.h $S/rbc/edg/imp.h
$B/rbc/force/area_volume/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/math/dev.h $S/msg.h $S/rbc/force/area_volume/dev/main.h $S/rbc/force/area_volume/imp.h $S/rbc/force/area_volume/imp/main.h $S/utils/cc.h $S/utils/kl.h $S/utils/texo.dev.h $S/utils/texo.h
$B/rbc/force/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/math/dev.h $S/msg.h $S/rbc/adj/dev.h $S/rbc/adj/type/common.h $S/rbc/adj/type/dev.h $S/rbc/force/area_volume/imp.h $S/rbc/force/dev/common.h $S/rbc/force/dev/main.h $S/rbc/force/dev/rnd0/main.h $S/rbc/force/dev/rnd1/main.h $S/rbc/force/dev/stress_free0/force.h $S/rbc/force/dev/stress_free0/shape.h $S/rbc/force/dev/stress_free1/force.h $S/rbc/force/dev/stress_free1/shape.h $S/rbc/force/imp.h $S/rbc/force/imp/fin.h $S/rbc/force/imp/forces.h $S/rbc/force/imp/ini.h $S/rbc/force/params/area_volume.h $S/rbc/force/params/lina.h $S/rbc/force/params/test.h $S/rbc/rnd/api/imp.h $S/rbc/rnd/imp.h $S/rbc/rnd/type.h $S/rbc/type.h $S/utils/cc.h $S/utils/kl.h $S/utils/te.h $S/utils/texo.dev.h $S/utils/texo.h
$B/rbc/gen/imp.o: $B/conf.h $S/inc/conf.h $S/inc/def.h $S/inc/type.h $S/io/off.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/rbc/gen/imp.h $S/rbc/gen/imp/main.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h
$B/rbc/main/anti/imp.o: $S/msg.h $S/rbc/adj/imp.h $S/rbc/adj/type/common.h $S/rbc/adj/type/hst.h $S/rbc/edg/imp.h $S/rbc/main/anti/imp.h $S/utils/error.h $S/utils/halloc.h
$B/rbc/main/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/io/off.h $S/io/restart.h $S/mpi/glb.h $S/mpi/wrapper.h $S/msg.h $S/rbc/adj/imp.h $S/rbc/adj/type/common.h $S/rbc/adj/type/hst.h $S/rbc/gen/imp.h $S/rbc/main/anti/imp.h $S/rbc/main/imp.h $S/rbc/main/imp/fin.h $S/rbc/main/imp/generate.h $S/rbc/main/imp/ini.h $S/rbc/main/imp/setup.h $S/rbc/main/imp/start.h $S/rbc/main/imp/util.h $S/rbc/type.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/mc.h
$B/rbc/rnd/api/imp.o: $B/conf.h $S/inc/conf.h $S/rbc/rnd/api/imp.h $S/rbc/rnd/api/imp/cpu.h $S/rbc/rnd/api/imp/cuda.h $S/rbc/rnd/api/imp/gaussrand.h $S/rbc/rnd/api/type.h $S/utils/error.h $S/utils/halloc.h
$B/rbc/rnd/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/mpi/glb.h $S/msg.h $S/rbc/rnd/api/imp.h $S/rbc/rnd/api/type.h $S/rbc/rnd/imp.h $S/rbc/rnd/imp/cu.h $S/rbc/rnd/imp/main.h $S/rbc/rnd/imp/seed.h $S/rbc/rnd/type.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/os.h
$B/rbc/stretch/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/rbc/stretch/dev/main.h $S/rbc/stretch/imp.h $S/rbc/stretch/imp/main.h $S/rbc/stretch/imp/type.h $S/rbc/stretch/imp/util.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/kl.h
$B/scheme/force/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/glb/get.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/scheme/force/dev/4roller.h $S/scheme/force/dev/constant.h $S/scheme/force/dev/double_poiseuille.h $S/scheme/force/dev/shear.h $S/scheme/force/imp.h $S/scheme/force/imp/4roller.h $S/scheme/force/imp/constant.h $S/scheme/force/imp/double_poiseuille.h $S/scheme/force/imp/none.h $S/scheme/force/imp/shear.h $S/utils/cc.h $S/utils/kl.h $S/utils/mc.h
$B/scheme/move/imp.o: $B/conf.h $S/d/ker.h $S/d/q.h $S/glb/get.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/scheme/move/dev/euler.h $S/scheme/move/dev/main.h $S/scheme/move/dev/vv.h $S/scheme/move/imp.h $S/scheme/move/imp/main.h $S/utils/kl.h
$B/scheme/restrain/imp.o: $B/conf.h $S/inc/conf.h $S/inc/def.h $S/inc/type.h $S/msg.h $S/scheme/restrain/imp.h $S/scheme/restrain/imp/none.h $S/scheme/restrain/imp/rbc_vel.h $S/scheme/restrain/imp/red_vel.h $S/scheme/restrain/sub/imp.h
$B/scheme/restrain/sub/imp.o: $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/scheme/restrain/sub/dev/color/map.h $S/scheme/restrain/sub/dev/dec.h $S/scheme/restrain/sub/dev/grey/map.h $S/scheme/restrain/sub/dev/main.h $S/scheme/restrain/sub/dev/main0.h $S/scheme/restrain/sub/dev/util.h $S/scheme/restrain/sub/imp.h $S/scheme/restrain/sub/imp/color/main.h $S/scheme/restrain/sub/imp/common.h $S/scheme/restrain/sub/imp/grey/main.h $S/scheme/restrain/sub/imp/main0.h $S/scheme/restrain/sub/stat/imp.h $S/scheme/restrain/sub/sum/imp.h $S/utils/cc.h $S/utils/kl.h
$B/scheme/restrain/sub/stat/imp.o: $S/scheme/restrain/sub/stat/imp.h $S/scheme/restrain/sub/stat/imp/dec.h $S/scheme/restrain/sub/stat/imp/main.h
$B/scheme/restrain/sub/sum/imp.o: $B/conf.h $S/inc/conf.h $S/mpi/wrapper.h $S/scheme/restrain/sub/sum/imp.h $S/scheme/restrain/sub/sum/imp/main.h $S/utils/mc.h
$B/u/rbc/main/lib/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/io/diag.h $S/io/mesh/imp.h $S/mpi/glb.h $S/msg.h $S/rbc/force/area_volume/imp.h $S/rbc/force/imp.h $S/rbc/main/imp.h $S/rbc/rnd/imp.h $S/rbc/stretch/imp.h $S/rbc/type.h $S/scheme/move/imp.h $S/u/rbc/main/lib/imp.h $S/u/rbc/main/lib/imp/main.h $S/utils/cc.h $S/utils/error.h $S/utils/halloc.h $S/utils/te.h $S/utils/texo.h
$B/u/rbc/main/main.o: $S/mpi/glb.h $S/u/rbc/main/lib/imp.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/msg.h $S/utils/cc/common.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/mc.h
$B/utils/os.o: $S/utils/os.h
