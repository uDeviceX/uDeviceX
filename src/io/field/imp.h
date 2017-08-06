bool H5FieldDump::directory_exists = false;
void H5FieldDump::header(FILE * xmf) {
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
}

void H5FieldDump::epilogue(FILE * xmf) {
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
}


H5FieldDump::H5FieldDump() : last_idtimestep(0) {
    const int L[3] = { XS, YS, ZS };

    for(int c = 0; c < 3; ++c)
    globalsize[c] = L[c] * m::dims[c];
}

void H5FieldDump::scalar(float *data,
                                   const char *channelname) {
    char path2h5[512];
    sprintf(path2h5, DUMP_BASE "/h5/%s.h5", channelname);
    fields(path2h5, &data, &channelname, 1);
}
