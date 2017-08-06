static void wrapper(const char* path, const char * const * const names, int n) {
    char w[BUFSIZ];
    FILE *f;
    sprintf(w, "%s.xmf", std::string(path).substr(0, std::string(path).find_last_of(".h5") - 2).data());
    f = fopen(w, "w");
    header(f);
    grid(f, std::string(path).substr(std::string(path).find_last_of("/") + 1).c_str(), names, n);
    epilogue(f);
    fclose(f);
}
