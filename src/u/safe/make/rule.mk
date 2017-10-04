$B/algo/minmax.o: $S/algo/minmax.cu; $N -I$S/. -I$S/algo
$B/algo/scan/imp.o: $S/algo/scan/imp.cu; $N -I$S/. -I$S/algo/scan
$B/clist/imp.o: $S/clist/imp.cu; $N -I$S/. -I$S/clist
$B/cnt/imp.o: $S/cnt/imp.cu; $N -I$S/. -I$S/cnt
$B/comm/imp.o: $S/comm/imp.cu; $N -I$S/. -I$S/comm
$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/d
$B/dbg/imp.o: $S/dbg/imp.cu; $N -I$S/. -I$S/dbg
$B/distr/flu/imp.o: $S/distr/flu/imp.cu; $N -I$S/. -I$S/distr/flu
$B/distr/rbc/imp.o: $S/distr/rbc/imp.cu; $N -I$S/. -I$S/distr/rbc
$B/distr/rig/imp.o: $S/distr/rig/imp.cu; $N -I$S/. -I$S/distr/rig
$B/dpdr/imp.o: $S/dpdr/imp.cu; $N -I$S/. -I$S/dpdr
$B/dpdr/int.o: $S/dpdr/int.cpp; $X -I$S/. -I$S/dpdr
$B/exch/flu/imp.o: $S/exch/flu/imp.cu; $N -I$S/. -I$S/exch/flu
$B/exch/mesh/imp.o: $S/exch/mesh/imp.cu; $N -I$S/. -I$S/exch/mesh
$B/exch/obj/imp.o: $S/exch/obj/imp.cu; $N -I$S/. -I$S/exch/obj
$B/field.o: $S/field.cpp; $X -I$S/.
$B/flu/imp.o: $S/flu/imp.cu; $N -I$S/. -I$S/flu
$B/frag/imp.o: $S/frag/imp.cpp; $X -I$S/. -I$S/frag
$B/fsi/imp.o: $S/fsi/imp.cu; $N -I$S/. -I$S/fsi
$B/glb.o: $S/glb.cu; $N -I$S/.
$B/hforces/imp.o: $S/hforces/imp.cu; $N -I$S/. -I$S/hforces
$B/inter/imp.o: $S/inter/imp.cu; $N -I$S/. -I$S/inter -I$S/inter/_ussr
$B/io/bop/imp.o: $S/io/bop/imp.cpp; $X -I$S/. -I$S/io/bop
$B/io/diag.o: $S/io/diag.cpp; $X -I$S/. -I$S/io
$B/io/field.o: $S/io/field.cpp; $X -I$S/. -I$S/io
$B/io/fields_grid.o: $S/io/fields_grid.cpp; $X -I$S/. -I$S/io
$B/io/mesh.o: $S/io/mesh.cpp; $X -I$S/. -I$S/io
$B/io/off.o: $S/io/off.cpp; $X -I$S/. -I$S/io
$B/io/ply.o: $S/io/ply.cpp; $X -I$S/. -I$S/io
$B/io/restart.o: $S/io/restart.cpp; $X -I$S/. -I$S/io
$B/io/rig.o: $S/io/rig.cpp; $X -I$S/. -I$S/io
$B/lforces/imp.o: $S/lforces/imp.cu; $N -I$S/. -I$S/lforces
$B/lforces/transpose/imp.o: $S/lforces/transpose/imp.cu; $N -I$S/. -I$S/lforces/transpose
$B/main.o: $S/main.cu; $N -I$S/.
$B/math/linal.o: $S/math/linal.cpp; $X -I$S/. -I$S/math
$B/mbounce/imp.o: $S/mbounce/imp.cu; $N -I$S/. -I$S/mbounce -I$S/mbounce/_safe
$B/mesh/bbox.o: $S/mesh/bbox.cu; $N -I$S/. -I$S/mesh
$B/mesh/collision.o: $S/mesh/collision.cu; $N -I$S/. -I$S/mesh
$B/mesh/dist.o: $S/mesh/dist.cpp; $X -I$S/. -I$S/mesh
$B/mesh/props.o: $S/mesh/props.cpp; $X -I$S/. -I$S/mesh
$B/meshbb/imp.o: $S/meshbb/imp.cu; $N -I$S/. -I$S/meshbb
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/mpi
$B/mpi/type.o: $S/mpi/type.cpp; $X -I$S/. -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/mpi
$B/mrescue.o: $S/mrescue.cu; $N -I$S/.
$B/msg.o: $S/msg.cpp; $X -I$S/.
$B/rbc/imp.o: $S/rbc/imp.cu; $N -I$S/. -I$S/rbc
$B/rbc/int.o: $S/rbc/int.cu; $N -I$S/. -I$S/rbc
$B/restrain/imp.o: $S/restrain/imp.cu; $N -I$S/. -I$S/restrain
$B/rig/imp.o: $S/rig/imp.cu; $N -I$S/. -I$S/rig
$B/rig/int.o: $S/rig/int.cu; $N -I$S/. -I$S/rig
$B/rigid/imp.o: $S/rigid/imp.cu; $N -I$S/. -I$S/rigid -I$S/rigid/_cuda
$B/rnd/imp.o: $S/rnd/imp.cpp; $X -I$S/. -I$S/rnd
$B/scheme/imp.o: $S/scheme/imp.cu; $N -I$S/. -I$S/scheme
$B/sdf/imp.o: $S/sdf/imp.cu; $N -I$S/. -I$S/sdf
$B/sdf/int.o: $S/sdf/int.cu; $N -I$S/. -I$S/sdf
$B/sim/imp.o: $S/sim/imp.cu; $N -I$S/. -I$S/sim
$B/tcells/imp.o: $S/tcells/imp.cu; $N -I$S/. -I$S/tcells
$B/tcells/int.o: $S/tcells/int.cpp; $X -I$S/. -I$S/tcells
$B/utils/cc.o: $S/utils/cc.cpp; $X -I$S/. -I$S/utils
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/utils
$B/utils/os.o: $S/utils/os.cpp; $X -I$S/. -I$S/utils
$B/vcontroller/imp.o: $S/vcontroller/imp.cu; $N -I$S/. -I$S/vcontroller
$B/wall/exch/imp.o: $S/wall/exch/imp.cpp; $X -I$S/. -I$S/wall/exch
$B/wall/force/imp.o: $S/wall/force/imp.cu; $N -I$S/. -I$S/wall/force
$B/wall/imp.o: $S/wall/imp.cu; $N -I$S/. -I$S/wall
$B/wall/int.o: $S/wall/int.cu; $N -I$S/. -I$S/wall
