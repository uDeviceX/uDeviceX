$B/conf/imp.o: $S/conf/imp.cpp; $X -I$S/. -I$S/conf
$B/coords/conf.o: $S/coords/conf.cpp; $X -I$S/. -I$S/coords
$B/coords/imp.o: $S/coords/imp.cpp; $X -I$S/. -I$S/coords
$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/d
$B/io/field/h5/imp.o: $S/io/field/h5/imp.cpp; $X -I$S/. -I$S/io/field/h5
$B/io/field/imp.o: $S/io/field/imp.cpp; $X -I$S/. -I$S/io/field
$B/io/field/xmf/imp.o: $S/io/field/xmf/imp.cpp; $X -I$S/. -I$S/io/field/xmf
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/mpi
$B/mpi/type.o: $S/mpi/type.cpp; $X -I$S/. -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/mpi
$B/u/io/h5/main.o: $S/u/io/h5/main.cpp; $X -I$S/.
$B/utils/cc.o: $S/utils/cc.cpp; $X -I$S/. -I$S/utils
$B/utils/error.o: $S/utils/error.cpp; $X -I$S/. -I$S/utils
$B/utils/imp.o: $S/utils/imp.cpp; $X -I$S/. -I$S/utils
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/utils
$B/utils/msg.o: $S/utils/msg.cpp; $X -I$S/. -I$S/utils
$B/utils/os.o: $S/utils/os.cpp; $X -I$S/. -I$S/utils
