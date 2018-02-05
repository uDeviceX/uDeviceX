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

void time_step_ini(Config *c, /**/ TimeStep **pq) {
    const char *type;
    TimeStep *q;
    EMALLOC(1, &q);
    UC(conf_lookup_string(c, "time.type", &type));
    if      (same_str(type, "const")) {
        q->type = CONST;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
    } else if (same_str(type, "disp")) {
        q->type = DISP;
        UC(conf_lookup_float(c, "time.dt", &q->dt));
        UC(conf_lookup_float(c, "time.dx", &q->dx));
    } else
        ERR("unrecognised type <%s>", type);
    *pq = q;
}
void time_step_fin(TimeStep *q) {EFREE(q); }

static float const_dt(TimeStep *q, MPI_Comm, TimeStepAccel*) { return q->dt; }
static float disp_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    float dt, dx, dt_max, accel;
    dt_max = q->dt;
    dx     = q->dx;
    accel = accel_max(comm, a);
    if (accel == 0) return dt_max;
    dt = 2*dx/(accel*accel);
    return dt > dt_max ? dt_max : dt;
}
float time_step_dt(TimeStep *q, MPI_Comm comm, TimeStepAccel *a) {
    return dt[q->type](q, comm, a);
}
