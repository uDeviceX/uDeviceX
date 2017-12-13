void run_eq(long te) { /* equilibrate */
    BForce fpar;
    ini_none(/**/ &fpar);    
    bool wall0 = false;
    for (long it = 0; it < te; ++it) step(&fpar, wall0, it);
    UC(distribute_flu(/**/ &flu));
}

void run(long ts, long te) {
    long it; /* current timestep */
    dump_strt_templ(&wall); /* :TODO: is it the right place? */

    BForce fpar;
    // TODO
#if   defined(FORCE_NONE)
    ini_none(/**/ &fpar);
#elif defined(FORCE_CONSTANT)
    BForce_cste_v par;
    float ex, ey, ez;
    os::env2float_d("FORCE_PAR_EX", 1, &ex);
    os::env2float_d("FORCE_PAR_EY", 0, &ey);
    os::env2float_d("FORCE_PAR_EZ", 0, &ez);
    par.a = make_float3(FORCE_PAR_A*ex, FORCE_PAR_A*ey, FORCE_PAR_A*ez);
    ini(par, /**/ &fpar);
#elif defined(FORCE_DOUBLE_POISEUILLE)
    BForce_dp_v par;
    par.a = FORCE_PAR_A;
    ini(par, /**/ &fpar);    
#elif defined(FORCE_SHEAR)
    BForce_shear_v par;
    par.a = FORCE_PAR_A;
    ini(par, /**/ &fpar);    
#elif defined(FORCE_4ROLLER)
    BForce_rol_v par;
    par.a = FORCE_PAR_A;
    ini(par, /**/ &fpar);    
#elif defined(FORCE_RADIAL)
    BForce_rad_v par;
    par.a = FORCE_PAR_A;
    ini(par, /**/ &fpar);    
#else
#error FORCE_* is undefined
#endif
    
    /* ts, te: time start and end */
    for (it = ts; it < te; ++it) {
        step2params(it - ts, &wall.vel, /**/ &wall.vview);
        step(&fpar, walls, it);
    }
    UC(distribute_flu(/**/ &flu));
}
