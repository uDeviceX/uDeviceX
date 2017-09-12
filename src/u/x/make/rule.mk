$B/algo/minmax.o: $S/algo/minmax.cu; $N -I$S/. -I$S/_rain -I$S/algo
$B/algo/scan/imp.o: $S/algo/scan/imp.cu; $N -I$S/. -I$S/_rain -I$S/algo/scan
$B/clist/imp.o: $S/clist/imp.cu; $N -I$S/. -I$S/_rain -I$S/clist
$B/cnt/imp.o: $S/cnt/imp.cu; $N -I$S/. -I$S/_rain -I$S/cnt
$B/comm/imp.o: $S/comm/imp.cu; $N -I$S/. -I$S/_rain -I$S/comm
$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/_rain -I$S/d
$B/dbg/imp.o: $S/dbg/imp.cu; $N -I$S/. -I$S/_rain -I$S/dbg
$B/distr/flu/imp.o: $S/distr/flu/imp.cu; $N -I$S/. -I$S/_rain -I$S/distr/flu
$B/dpdr/imp.o: $S/dpdr/imp.cu; $N -I$S/. -I$S/_rain -I$S/dpdr
$B/dpdr/int.o: $S/dpdr/int.cpp; $X -I$S/. -I$S/_rain -I$S/dpdr
$B/dual/imp.o: $S/dual/imp.cpp; $X -I$S/. -I$S/_rain -I$S/dual
$B/field.o: $S/field.cpp; $X -I$S/. -I$S/_rain
$B/flu/imp.o: $S/flu/imp.cu; $N -I$S/. -I$S/_rain -I$S/flu -I$S/flu/_ussr
$B/flu/int.o: $S/flu/int.cu; $N -I$S/. -I$S/_rain -I$S/flu -I$S/flu/_ussr
$B/fsi/imp.o: $S/fsi/imp.cu; $N -I$S/. -I$S/_rain -I$S/fsi
$B/glb.o: $S/glb.cu; $N -I$S/. -I$S/_rain
$B/hforces/imp.o: $S/hforces/imp.cu; $N -I$S/. -I$S/_rain -I$S/hforces
$B/inter/imp.o: $S/inter/imp.cu; $N -I$S/. -I$S/_rain -I$S/inter
$B/io/bop/imp.o: $S/io/bop/imp.cpp; $X -I$S/. -I$S/_rain -I$S/io/bop
$B/io/diag.o: $S/io/diag.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/field.o: $S/io/field.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/fields_grid.o: $S/io/fields_grid.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/off.o: $S/io/off.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/ply.o: $S/io/ply.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/rbc.o: $S/io/rbc.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/restart.o: $S/io/restart.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/io/rig.o: $S/io/rig.cpp; $X -I$S/. -I$S/_rain -I$S/io
$B/lforces/local.o: $S/lforces/local.cu; $N -I$S/. -I$S/_rain -I$S/lforces
$B/main.o: $S/main.cu; $N -I$S/. -I$S/_rain
$B/math/linal.o: $S/math/linal.cpp; $X -I$S/. -I$S/_rain -I$S/math
$B/mbounce/imp.o: $S/mbounce/imp.cu; $N -I$S/. -I$S/_rain -I$S/mbounce -I$S/mbounce/_release
$B/mcomm/imp.o: $S/mcomm/imp.cu; $N -I$S/. -I$S/_rain -I$S/mcomm
$B/mcomm/int.o: $S/mcomm/int.cu; $N -I$S/. -I$S/_rain -I$S/mcomm
$B/mdstr/imp.o: $S/mdstr/imp.cu; $N -I$S/. -I$S/_rain -I$S/mdstr
$B/mdstr/int.o: $S/mdstr/int.cpp; $X -I$S/. -I$S/_rain -I$S/mdstr
$B/mesh/bbox.o: $S/mesh/bbox.cu; $N -I$S/. -I$S/_rain -I$S/mesh
$B/mesh/collision.o: $S/mesh/collision.cu; $N -I$S/. -I$S/_rain -I$S/mesh
$B/mesh/dist.o: $S/mesh/dist.cpp; $X -I$S/. -I$S/_rain -I$S/mesh
$B/mesh/props.o: $S/mesh/props.cpp; $X -I$S/. -I$S/_rain -I$S/mesh
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/_rain -I$S/mpi
$B/mpi/type.o: $S/mpi/type.cpp; $X -I$S/. -I$S/_rain -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/_rain -I$S/mpi
$B/mrescue.o: $S/mrescue.cu; $N -I$S/. -I$S/_rain
$B/msg.o: $S/msg.cpp; $X -I$S/. -I$S/_rain
$B/odstr/halo/imp.o: $S/odstr/halo/imp.cu; $N -I$S/. -I$S/_rain -I$S/odstr/halo -I$S/odstr/halo/_release
$B/odstr/imp.o: $S/odstr/imp.cu; $N -I$S/. -I$S/_rain -I$S/odstr -I$S/odstr/_release
$B/odstr/int.o: $S/odstr/int.cu; $N -I$S/. -I$S/_rain -I$S/odstr -I$S/odstr/_release
$B/rbc/imp.o: $S/rbc/imp.cu; $N -I$S/. -I$S/_rain -I$S/rbc
$B/rbc/int.o: $S/rbc/int.cu; $N -I$S/. -I$S/_rain -I$S/rbc
$B/rdstr/imp.o: $S/rdstr/imp.cu; $N -I$S/. -I$S/_rain -I$S/rdstr
$B/rdstr/int.o: $S/rdstr/int.cu; $N -I$S/. -I$S/_rain -I$S/rdstr
$B/rex/imp.o: $S/rex/imp.cu; $N -I$S/. -I$S/_rain -I$S/rex -I$S/rex/_release
$B/rex/int.o: $S/rex/int.cpp; $X -I$S/. -I$S/_rain -I$S/rex -I$S/rex/_release
$B/rig/imp.o: $S/rig/imp.cu; $N -I$S/. -I$S/_rain -I$S/rig
$B/rig/int.o: $S/rig/int.cu; $N -I$S/. -I$S/_rain -I$S/rig
$B/rigid/imp.o: $S/rigid/imp.cu; $N -I$S/. -I$S/_rain -I$S/rigid -I$S/rigid/_cuda
$B/rnd/imp.o: $S/rnd/imp.cpp; $X -I$S/. -I$S/_rain -I$S/rnd
$B/scheme/imp.o: $S/scheme/imp.cu; $N -I$S/. -I$S/_rain -I$S/scheme
$B/sdf/imp.o: $S/sdf/imp.cu; $N -I$S/. -I$S/_rain -I$S/sdf
$B/sdf/int.o: $S/sdf/int.cu; $N -I$S/. -I$S/_rain -I$S/sdf
$B/sim/imp.o: $S/sim/imp.cu; $N -I$S/. -I$S/_rain -I$S/sim
$B/tcells/imp.o: $S/tcells/imp.cu; $N -I$S/. -I$S/_rain -I$S/tcells
$B/tcells/int.o: $S/tcells/int.cpp; $X -I$S/. -I$S/_rain -I$S/tcells
$B/utils/cc.o: $S/utils/cc.cpp; $X -I$S/. -I$S/_rain -I$S/utils
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/_rain -I$S/utils
$B/utils/os.o: $S/utils/os.cpp; $X -I$S/. -I$S/_rain -I$S/utils
$B/wall/exch/imp.o: $S/wall/exch/imp.cpp; $X -I$S/. -I$S/_rain -I$S/wall/exch
$B/wall/imp.o: $S/wall/imp.cu; $N -I$S/. -I$S/_rain -I$S/wall
$B/wall/int.o: $S/wall/int.cu; $N -I$S/. -I$S/_rain -I$S/wall
