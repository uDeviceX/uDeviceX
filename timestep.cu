#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE

#include "config.cuh"
#include "macros.cuh"

#include <math.h>
#include <mpi.h>

struct Config;
struct Force;

void emalloc(size_t, void **);
void efree(void *);
bool same_str(const char *, const char *);
void msg_print(const char *, ...);
float force_stat_max(int, const Force *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)
namespace mpicheck {
void check(int, const char *, int);
}

namespace m {
int Reduce(const void *, void *, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
bool is_master(MPI_Comm);
}

void conf_lookup_string(const Config *, const char *, const char **);
void conf_lookup_float(const Config *, const char *, float *);
void conf_lookup_bool(const Config *, const char *, int *);

enum { MAX_ACCEL_NUM = 9 };

struct TimeStepAccel {
    float mm[MAX_ACCEL_NUM];
    int nn[MAX_ACCEL_NUM];
    Force *fff[MAX_ACCEL_NUM];
    int k;
};

struct TimeStep {
    int type;
    bool screenlog;
    float dt;
    float dx;
    float accel[MAX_ACCEL_NUM];
    int k;
};

enum { CONST, DISP };
static float const_dt(TimeStep *, MPI_Comm, TimeStepAccel *);
static float disp_dt(TimeStep *, MPI_Comm, TimeStepAccel *);
static void const_log(TimeStep *);
static void disp_log(TimeStep *);
static void (*tlog[])(TimeStep *) = {const_log, disp_log};
static float (*dt[])(TimeStep *, MPI_Comm, TimeStepAccel *) = {const_dt, disp_dt};

static void reduce(MPI_Comm comm, const void *sendbuf0, void *recvbuf,
                   int count, MPI_Datatype datatype, MPI_Op op) {
    const void *sendbuf = (m::is_master(comm) ? MPI_IN_PLACE : sendbuf0);
    MC(m::Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm));
}

static float accel_max(MPI_Comm comm, TimeStepAccel *q, float *accel) {
    float max = 0;
    for (int i = 0; i < q->k; i++) {
        float force = force_stat_max(q->nn[i], q->fff[i]);
        accel[i] = force / q->mm[i];
        if (accel[i] > max) max = accel[i];
    }
    UC(reduce(comm, &max, &max, 1, MPI_FLOAT, MPI_MAX));
    return max;
}

void time_step_accel_ini(TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    q->k = 0;
    *pq = q;
}

void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }

void time_step_accel_push(TimeStepAccel *q, float m, int n, Force *ff) {
    int k = q->k;
    if (k + 1 == MAX_ACCEL_NUM)
        ERR("k + 1 = %d < %d = MAX_ACCEL_NUM", k + 1, MAX_ACCEL_NUM);
    if (m <= 0) ERR("m <= 0");
    q->mm[k] = m;
    q->nn[k] = n;
    q->fff[k] = ff;
    q->k = k + 1;
}

void time_step_accel_reset(TimeStepAccel *q) { q->k = 0; }

void time_step_ini(const Config *c, TimeStep **pq) {
    const char *type;
    int screenlog;
    TimeStep *q;
    EMALLOC(1, &q);

    UC(conf_lookup_string(c, "time.type", &type));
    if (same_str(type, "const")) {
        q->type = CONST;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
    } else if (same_str(type, "disp")) {
        q->type = DISP;
        q->k = 0;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
        UC(conf_lookup_float(c, "time.dx", &q->dx));
    } else {
        ERR("unrecognised time.type: '%s'", type);
    }
    UC(conf_lookup_bool(c, "time.screenlog", &screenlog));
    q->screenlog = screenlog;
    *pq = q;
}

void time_step_fin(TimeStep *q) { EFREE(q); }

static float const_dt(TimeStep *q, MPI_Comm, TimeStepAccel *) { return q->dt; }

static float disp_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    float dt_max = q->dt;
    float accel = accel_max(comm, a, q->accel);
    q->k = a->k;
    if (accel == 0) return dt_max;
    float dt_new = sqrtf(2. * q->dx / accel);
    return dt_new > dt_max ? dt_max : dt_new;
}

float time_step_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    return dt[q->type](q, comm, a);
}

static void const_log(TimeStep *q) {
    msg_print(" time_step: const: dt=%.4g\n", q->dt);
}

static void disp_log(TimeStep *q) {
    msg_print(" time_step: disp: dt=%.4e, dx=%.4e", q->dt, q->dx);
    for (int i = 0; i < q->k; i++) {
        float dt_acc = sqrtf(2. * q->dx / q->accel[i]);
        msg_print(" accel:%.4e, dt_acc:%.4e, dt_used:%.4e",
                  q->accel[i], dt_acc, dt_acc > q->dt ? q->dt : dt_acc);
    }
    msg_print("");
}

void time_step_log(TimeStep *q) {
    if (q->screenlog == true) tlog[q->type](q);
}

float time_step_dt0(TimeStep *q) { return q->dt; }
