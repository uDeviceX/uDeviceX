enum { MAX_ACCEL_NUM = 9 };
struct TimeStepAccel {
    float  mm[MAX_ACCEL_NUM];
    int    nn[MAX_ACCEL_NUM];
    Force *fff[MAX_ACCEL_NUM];
    int k;
};

struct TimeStep {
    int type;
    float dt;
    float dx;

    /* to log */
    float accel[MAX_ACCEL_NUM];
    int k;
};

enum {CONST, DISP};
static float const_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
static float disp_dt(TimeStep*, MPI_Comm, TimeStepAccel*);
static void const_log(TimeStep*);
static void  disp_log(TimeStep*);
static void  (*log[])(TimeStep*)                          = { const_log,  disp_log };
static float (*dt[])(TimeStep*, MPI_Comm, TimeStepAccel*) = { const_dt,  disp_dt };
