#include <cmath>
#include <cassert>
#include <chrono>
#include <random>
#include <iterator>
#include <ctime>
#include <iomanip>
#include <stdio.h>
#include <cstdlib> 
#include <queue>
#include "graph.hpp"

#ifndef HELPER_HPP
#define HELPER_HPP

using namespace std;

class Helper {
public:
    HyperGraph *graph;
    vector<bool> hypergraph_masking;
    vector<bool> node_masking;
    set<string> check_pair;
    vector<string> tmp;
    vector<int> check;

    bool cch_flag = false;
    bool egonet_flag = false;
    string dataname;
    string outputdir;

    Helper(set<int> subhypergraph, HyperGraph *graph, string outputdir, string algo_opt);
    ~Helper(){
        hypergraph_masking.clear();
        node_masking.clear();
    }
    void get_clustering_coef_hedge(void);
    void get_egonet_prop(void);

    void update(set<int> deltaset, HyperGraph *graph);
    void save_properties(void);
};
#endif