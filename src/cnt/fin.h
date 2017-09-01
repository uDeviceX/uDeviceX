void fin() {
    delete indexes;
    scan::free_work(&ws);
    delete entries;
    delete rgen;

    Dfree(counts);
    Dfree(starts);
}
