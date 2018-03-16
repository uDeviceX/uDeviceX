static config_setting_t* subsetting(const char *desc, int type, config_setting_t *group) {
    config_setting_t *sub;
    sub = config_setting_lookup(group, desc);
    if (NULL == sub)
        sub = config_setting_add(group, desc, type);
    return sub;
}

static config_setting_t* get_subgroup_setting(int n, const char *desc[], config_t *c) {
    config_setting_t *group;
    group = config_root_setting(c);
    
    for (int i = 0; i < n; ++i)
        group = subsetting(desc[i], CONFIG_TYPE_GROUP, group);
    return group;    
}

void conf_set_int(int n, const char *desc[], int a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_INT, /**/ group);
    
    status = config_setting_set_int(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_vint(int n, const char *desc[], int nelem, const int a[], Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting, *status;
    int i;
    enum {APPEND = -1};
    c = &cfg->c[EXE];
        
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY, /**/ group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_int_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

void conf_set_int3(int n, const char *desc[], int3 a, Config *cfg) {
    const int a3[] = {a.x, a.y, a.z};
    UC(conf_set_vint(n, desc, 3, a3, cfg));
}

void conf_set_float(int n, const char *desc[], float a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_FLOAT, /**/ group);
    
    status = config_setting_set_float(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_vfloat(int n, const char *desc[], int nelem, const float a[], Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting, *status;
    int i;
    enum {APPEND = -1};
    c = &cfg->c[EXE];
        
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_ARRAY, /**/ group);

    for (i = 0; i < nelem; ++i) {
        status = config_setting_set_float_elem(setting, APPEND, a[i]);
        if (NULL == status)
            ERR("could not set element %d/%d of <%s>", i, nelem, desc[n-1]);
    }
}

void conf_set_float3(int n, const char *desc[], float3 a, Config *cfg) {
    const float a3[] = {a.x, a.y, a.z};
    UC(conf_set_vfloat(n, desc, 3, a3, cfg));
}

void conf_set_bool(int n, const char *desc[], int a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_BOOL, /**/ group);
    
    status = config_setting_set_bool(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}

void conf_set_string(int n, const char *desc[], const char *a, Config *cfg) {
    config_t *c;
    config_setting_t *group, *setting;
    int status;
    c = &cfg->c[EXE];
    
    group = get_subgroup_setting(n-1, desc, /**/ c);
    setting = subsetting(desc[n-1], CONFIG_TYPE_STRING, /**/ group);
    
    status = config_setting_set_string(setting, a);
    if (CONFIG_TRUE != status)
        ERR("could not set <%s>", desc[n-1]);
}
