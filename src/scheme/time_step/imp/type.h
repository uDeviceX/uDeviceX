enum { MAX_ACCEL_NUM = 9 };
struct TimeStepAccel {
    float  mm[MAX_ACCEL_NUM];
    int    nn[MAX_ACCEL_NUM];
    Force *fff[MAX_ACCEL_NUM];
    int k;
};

struct Const {
};

struct Disp {

};

enum {CONST, DISP};
float (*dt[]) = {NULL, NULL};

struct TimeStep {
    int type;
    void *q;
};

