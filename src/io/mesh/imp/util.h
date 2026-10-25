#pragma once

static char *mesh_cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *mesh_cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }

static void mkdir0(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    mesh_cpy(path, p);
    mesh_cat(path, "/");
    mesh_cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

static void mesh_mkdir(MPI_Comm comm, const char *p, const char *s) {
    if (m::is_master(comm)) UC(mkdir0(p, s));
}
