#pragma once

#include <vector>
#include <string>

struct BopData;

class PyBop {
public:
    PyBop();
    ~PyBop();

    void reset();
    
    void alloc();
    void set_n(long n);
    void set_vars(int n, const std::string &vars);
    void set_type(const std::string &type);

    long get_n();
    std::string get_vars();
    std::string get_type();

    void set_datai(const std::vector<int>    &data);
    void set_dataf(const std::vector<float>  &data);
    void set_datad(const std::vector<double> &data);

    std::vector<int>    get_datai();
    std::vector<float>  get_dataf();
    std::vector<double> get_datad();
   
    void write(std::string basename);
    void read(std::string hname);
    
private:
    BopData *d;
};
