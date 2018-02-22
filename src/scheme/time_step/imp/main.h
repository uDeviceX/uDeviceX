void time_step_accel_ini(/**/ TimeStepAccel **pq) {
    TimeStepAccel *q;
    EMALLOC(1, &q);
    q->k = 0;
    *pq = q;
}
void time_step_accel_fin(TimeStepAccel *q) { EFREE(q); }
void time_step_accel_push(TimeStepAccel *q, float m, int n, Force *ff) {
    int k;
    k = q->k;
    if (k + 1 == MAX_ACCEL_NUM)
        ERR("k + 1 = %d < %d = MAX_ACCEL_NUM", k + 1, MAX_ACCEL_NUM);
    if (m <= 0) ERR("m <= 0");
    q->mm[k] = m; q->nn[k] = n; q->fff[k] = ff;
    q->k = k + 1;
}
void time_step_accel_reset(TimeStepAccel *q) { q->k = 0; }

void time_step_ini(Config *c, /**/ TimeStep **pq) {
    const char *type;
    int l;
    TimeStep *q;
    EMALLOC(1, &q);

    UC(conf_lookup_string(c, "time.type", &type));
    if      (same_str(type, "const")) {
        q->type = CONST;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
    } else if (same_str(type, "disp")) {
        q->type = DISP;
        q->k    = 0;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
        UC(conf_lookup_float(c, "time.dx", &q->dx));
    } else
        ERR("unrecognised time.type: '%s'", type);

    UC(conf_lookup_bool(c, "time.screenlog", &l));
    q->screenlog = l;

    *pq = q;
}
void time_step_fin(TimeStep *q) {EFREE(q); }

static float const_dt(TimeStep *q, MPI_Comm, TimeStepAccel*) { return q->dt; }
static float disp_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    float dt_new, dx, dt_max, accel;
    dt_max = q->dt;
    dx     = q->dx;
    accel = accel_max(comm, a, /**/ q->accel);
    q->k  = a->k;
    if (accel == 0) return dt_max;
    dt_new = sqrtf(2.*dx/accel);
    return (dt_new > dt_max ? dt_max : dt_new);
}
float time_step_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    return dt[q->type](q, comm, a);
}

static void const_log(TimeStep *q) { msg_print(" time_step: const: dt=%.4g\n", q->dt); }
static void  disp_log(TimeStep *q) {
    int i;
    float accel;
    float dx;
    dx = q->dx;
    msg_print(" time_step: disp: dt=%.4e, dx=%.4e", q->dt, q->dx);
    for (i = 0; i < q->k; i++) {
        accel = q->accel[i];
        msg_print(" accel:%.4e, dt_acc:%.4e, dt_used:%.4e", accel, sqrtf(2.*dx/accel), (sqrtf(2.*dx/accel) > q->dt ? q->dt : sqrtf(2.*dx/accel)));
    }
    msg_print("");
}
void time_step_log(TimeStep *q) { if (q->screenlog==true) tlog[q->type](q); }
float time_step_dt0(TimeStep *q) { return q->dt; };
