static void wrapper(const char* const path, const char * const * const channelnames, const int nchannels) {
    char w[256];
    sprintf(w, "%s.xmf", std::string(path).substr(0, std::string(path).find_last_of(".h5") - 2).data());
    FILE * xmf = fopen(w, "w");
    header(xmf);
    grid(xmf, std::string(path).substr(std::string(path).find_last_of("/") + 1).c_str(), channelnames, nchannels);
    epilogue(xmf);
    fclose(xmf);
}
