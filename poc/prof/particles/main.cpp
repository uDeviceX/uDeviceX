// CUDA runtime
#include <cuda_runtime.h>

#include "particleSystem.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE       64
#define NUM_PARTICLES   16384

const uint width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;

int mode = 0;
bool displayEnabled = false;
bool bPause = false;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;

enum { M_VIEW = 0, M_MOVE };

uint numParticles = 0;
uint3 gridSize;
int numIterations = 0; // run until exit

// simulation parameters
float timestep = 0.5f;
float damping = 1.0f;
float gravity = 0.0003f;
int iterations = 1;
int ballr = 10;

float collideSpring = 0.5f;;
float collideDamping = 0.02f;;
float collideShear = 0.1f;
float collideAttraction = 0.0f;

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

float modelView[16];

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize)
{
    psystem = new ParticleSystem(numParticles, gridSize);
    psystem->reset(ParticleSystem::CONFIG_GRID);
    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (psystem)
    {
        delete psystem;
    }
    return;
}

void runBenchmark(int iterations, char *exec_path)
{
    printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);
}


inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void addSphere()
{
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr+(pr*2.0f)*ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand()*(2.0f - tr*2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr*2.0f);
}

void initParams()
{
    timestep = 0.0f;
    damping = 0.0f;
    gravity = 0.0f;
    ballr = 1;
    collideSpring = 0.0f;
    collideDamping = 0.0f;
    collideShear = 0.0f;
    collideAttraction = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;
    numIterations = 0;

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
        {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "grid"))
        {
            gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
            fpsLimit = frameCheckNumber;
            numIterations = 1;
        }
    }

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("particles: %d\n", numParticles);

    bool benchmark = checkCmdLineFlag(argc, (const char **) argv, "benchmark") != 0;

    if (checkCmdLineFlag(argc, (const char **) argv, "i"))
    {
        numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
    }

    if (g_refFile)
    {
        cudaInit(argc, argv);
    }
    else
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    initParticleSystem(numParticles, gridSize);
    initParams();

    if (benchmark || g_refFile)
    {
        if (numIterations <= 0)
        {
            numIterations = 300;
        }

        runBenchmark(numIterations, argv[0]);
    }

    if (psystem)
    {
        delete psystem;
    }

    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

