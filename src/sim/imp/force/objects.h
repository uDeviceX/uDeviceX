void forces_objects(Sim *sim) {
    PFarrays *obj;
    PFarray flu;
    
    UC(pfarrays_ini(&obj));
    UC(objects_get_particles_all(sim->obj, obj));
    UC(utils_get_pf_flu(sim, &flu));
    
    UC(obj_inter_forces(sim->objinter, &flu, sim->flu.q.cells.starts, obj));

    UC(pfarrays_fin(obj));
}

