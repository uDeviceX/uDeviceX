namespace dump
{
    void init()
    {
        if (m::rank == 0)
        mkdir("bop", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    
    void parts(const Particle *pp, const long n, const char *name, const int step)
    {
        // serial for now
        if (m::rank != 0) exit(1);
        
        char fname[256] = {0};
        sprintf(fname, "bop/%s-%05d.bop", name, step);
        
        FILE *f = fopen(fname, "w");

        if (f == NULL)
        {
            fprintf(stderr, "(dump) error opening <%s>\n", fname);
            exit(1);
        }

        long nl = n;
        fwrite(&nl, sizeof(long), 1, f);
        fwrite(pp, sizeof(Particle), n, f);
        
        fclose(f);
    }

}
