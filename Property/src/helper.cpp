#include "helper.hpp"

Helper::Helper(set<int> subhypergraph, HyperGraph *graph, string outputdir, string algo_opt) {
    this->graph = graph;
    this->hypergraph_masking.resize(graph->number_of_hedges, false);
    this->node_masking.resize(graph->number_of_nodes, false);
    this->outputdir = outputdir;
    
    // setting flag
    if ((int)algo_opt.size() == 0){
        cch_flag = true;
        egonet_flag = true;
    }
    else{
        tmp = split(algo_opt, ',');
        for (int i = 0 ; i < (int)tmp.size() ; i++){
            if (tmp[i].compare("clusteringcoef_hedge") == 0) cch_flag = true;
            else if (tmp[i].compare("egonet") == 0) egonet_flag = true;
        }
    }

    // update by subhypergraph
    if ((int)subhypergraph.size() > 0){
        set<int> nodeset;
        for (int h : subhypergraph){
            fill(check.begin(), check.end(), 0);
            this->hypergraph_masking[h] = true;
            int hsize = (int)graph->hyperedge2node[h].size();
            for (int i = 0; i < hsize ; i++){
                int vi = graph->hyperedge2node[h][i];
                nodeset.insert(vi);
                if (!this->node_masking[vi]){
                    this->node_masking[vi] = true; // update node_masking
                }
            }
        }
    }
}

void Helper::get_clustering_coef_hedge(void){
    vector<bool> check_hedges;
    vector<bool> check_tmp;
    check_hedges.resize(graph->number_of_hedges, false);
    check_tmp.resize(graph->number_of_hedges, false);
    // set<int> check_hedges;

    string writeFile2 = outputdir + "clusteringcoef_hedge.txt";
    ofstream resultFile2(writeFile2.c_str(), fstream::out);

    for (int v = 0; v < (int)node_masking.size() ; v++){
        int vdeg = (int)graph->node2hyperedge[v].size();
        // check_pair.clear();
        // check_hedges.clear();
        fill(check_hedges.begin(), check_hedges.end(), false);

        for (int nhi = 0 ; nhi < vdeg ; nhi++){
            int nh = graph->node2hyperedge[v][nhi];
            if (hypergraph_masking[nh]){
                check_hedges[nh] = true;
            }
        }
        double cc = 0.0; // number of connected neighbor pairs
        for (int h=0 ; h < graph->number_of_hedges ; h++){
            if (!check_hedges[h]){
                continue;
            }
            fill(check_tmp.begin(), check_tmp.end(), false);
            int hsize = (int)graph->hyperedge2node[h].size();
            for (int si=0 ; si<hsize ; si++){
                int nv = graph->hyperedge2node[h][si];
                int nv_deg = (int)graph->node2hyperedge[nv].size();
                for (int di=0 ; di<nv_deg ; di++){
                    int nh = graph->node2hyperedge[nv][di];
                    if (check_hedges[nh]){
                        if ((h < nh) && (!check_tmp[nh])){
                            check_tmp[nh] = true;
                            cc += 1;
                        }
                    }
                }
            }
        }
        resultFile2 << to_string(vdeg) << "," << to_string(cc) << endl;
    }
    resultFile2.close();
    cout << "cc hedge" << endl;
}

void Helper::get_egonet_prop(void){
    double sum_of_hsizes = 0;
    double overlapness, density;
    set<int> nodeset;
    int numh = 0, numv = 0;

    string writeFile1 = outputdir + "density_dist.txt";
    ofstream resultFile1(writeFile1.c_str(),  fstream::out);
    string writeFile2 = outputdir + "overlapness_dist.txt";
    ofstream resultFile2(writeFile2.c_str(),  fstream::out);

    for (int v = 0; v < graph->number_of_nodes ; v++){
        int vdeg = graph->node2hyperedge[v].size();
        sum_of_hsizes = 0;
        numh = 0;
        nodeset.clear();

        for (int hi = 0 ; hi < vdeg ; hi++){
            int h = graph->node2hyperedge[v][hi];
            int hsize = graph->hyperedge2node[h].size();
            for (int nvi = 0; nvi < hsize; nvi++){
                int nv = graph->hyperedge2node[h][nvi];
                nodeset.insert(nv);
            }
            numh ++;
            sum_of_hsizes += hsize;
        }
        numv = (int)nodeset.size();
        density = (double)numh / numv;
        overlapness = (double)sum_of_hsizes / numv;
        resultFile1 << to_string(numv) << "," << to_string(numh) << endl;
        resultFile2 << to_string(numv) << "," << to_string(sum_of_hsizes) << endl;
    }
    resultFile1.close();
    resultFile2.close();

    cout << "egonet" << endl;
}

void Helper::update(set<int> deltaset, HyperGraph *graph){
    int count = 0;
    int step = max((int)deltaset.size() / 1000, 1);
    for (auto h : deltaset){
        count += 1;
        assert(!hypergraph_masking[h]);
        hypergraph_masking[h] = true;
        int hsize = (int)graph->hyperedge2node[h].size();

        for (int i = 0 ; i < hsize ; i++){
            int v = graph->hyperedge2node[h][i];
            if (!node_masking[v]){
                node_masking[v] = true;
            }
        }
    }
    return;
}

void Helper::save_properties(void){
    cout << "Get Egonet" << endl;
    if (egonet_flag) get_egonet_prop();
    if (cch_flag){
        cout << "Get Clustering Hedge" << endl;
        get_clustering_coef_hedge();
    }
}
