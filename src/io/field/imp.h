static void header(FILE * xmf) {
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
}

static void epilogue(FILE * xmf) {
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
}
