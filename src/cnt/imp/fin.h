void fin() {
    delete indexes;
    scan::free_work(&g::ws);
    delete entries;
    delete rgen;

    Dfree(g::counts);
    Dfree(g::starts);
}
