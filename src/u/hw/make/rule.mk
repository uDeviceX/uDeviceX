$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/d
$B/glb.o: $S/glb.cu; $N -I$S/.
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/mpi
$B/msg.o: $S/msg.cpp; $X -I$S/.
$B/u/hw/main.o: $S/u/hw/main.cu; $N -I$S/.
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/utils
