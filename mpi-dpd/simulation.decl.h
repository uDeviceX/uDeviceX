/**** from simulation.h ****/
ParticleArray *particles_pingpong[2];
ParticleArray *particles, *newparticles;
ParticleArray *rbcscoll;

SimpleDeviceBuffer<float4 > *xyzouvwo;
SimpleDeviceBuffer<ushort4> *xyzo_half;

CellLists* cells;
RedistributeRBCs* redistribute_rbcs;

ComputeDPD* dpd;
SoluteExchange* solutex;
ComputeFSI* fsi;
ComputeContact* contact;

bool wall_created = false;
bool sim_is_done = false;

MPI_Comm activecomm;
cudaStream_t mainstream, uploadstream, downloadstream;

size_t nsteps;
float driving_acceleration = 0;

pthread_t thread_datadump;
pthread_mutex_t mutex_datadump;
pthread_cond_t request_datadump, done_datadump;
bool datadump_pending = false;
int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs;
bool async_thread_initialized;

PinnedHostBuffer<Particle>      *particles_datadump;
PinnedHostBuffer<Acceleration>  *accelerations_datadump;

cudaEvent_t evdownloaded;

#define NPMAX 5000000 /* TODO: */
float rbc_xx[NPMAX], rbc_yy[NPMAX], rbc_zz[NPMAX];
float sol_xx[NPMAX], sol_yy[NPMAX], sol_zz[NPMAX];
int iotags[NPMAX];
