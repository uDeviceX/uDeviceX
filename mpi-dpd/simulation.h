/*
 *  simulation.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

void sim_init(MPI_Comm cartcomm_, MPI_Comm activecomm_);
void sim_run();
void sim_close();
