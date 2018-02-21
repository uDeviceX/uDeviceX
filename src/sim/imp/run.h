static float get_dt0(Sim *s) {
    TimeStep *time_step;    
    time_step = s->time_step;
    return time_step_dt0(time_step);
}

static float get_dt(Sim *s) {
    TimeStep *time_step;
    TimeStepAccel *accel;
    MPI_Comm comm;
    float dt;
    comm = s->cart;
    time_step = s->time_step;
    time_step_accel_ini(&accel);
    dt = time_step_dt(time_step, comm, accel);
    time_step_accel_fin(accel);
    return dt;
}

static void run_eq(Time *time, float te, Sim *s) { /* equilibrate */
    float dt;
    BForce *bforce;
    UC(bforce_ini(&bforce));
    s->equilibrating = true;
    bforce_ini_none(/**/ bforce);    
    bool wall0 = false;
    dt = get_dt0(s);    
    while (time_current(time) < te) {
        UC(step(time, dt, bforce, wall0, 0.0, s));
        time_next(time, dt);
        dt = get_dt(s);        
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}

static void run(const Config *cfg, Time *time, float ts, float te, Sim *s) {
    float dt;
    Wall *wall = &s->wall;
    BForce *bforce;

    UC(bforce_ini(&bforce));
    UC(bforce_ini_conf(cfg, bforce));

    dump_strt_templ(s->coords, wall, s); /* :TODO: is it the right place? */
    s->equilibrating = false;   

    /* ts, te: time start and end */
    dt = get_dt0(s);
    while (time_current(time) < te) {
        UC(step(time, dt, bforce, walls, ts, s));
        time_next(time, dt);
        dt = get_dt(s);
    }
    UC(distribute_flu(/**/ s));
    UC(bforce_fin(bforce));
}
