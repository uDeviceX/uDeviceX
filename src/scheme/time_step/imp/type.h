enum { MAX_ACCEL_NUM = 9 };
struct TimeStepAccel {
    float  mm[MAX_ACCEL_NUM];
    int    nn[MAX_ACCEL_NUM];
    Force *fff[MAX_ACCEL_NUM];
    int k;
};

struct Const { };
struct Disp  { };

struct TimeStep {
    int type;
    float dt;
    float dx;
};

enum {CONST, DISP};
static float disp_dt(MPI_Comm, TimeStepAccel*, TimeStep*);
static float const_dt(MPI_Comm, TimeStepAccel*, TimeStep*);
static float (*dt [])(MPI_Comm, TimeStepAccel*, TimeStep*) = { const_dt,  disp_dt };
