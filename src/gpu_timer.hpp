#include "ccs.hpp"


namespace GPU_TIMER {

#define EVALUATE 0
#define TABLE_LEN 7
#define DELTA_T 2.5
#define THREAD_NUM 512
#define BLOCK_NUM(n) (((n) + THREAD_NUM - 1) / THREAD_NUM)
const double deltaV = 0.01 * VDD;

#define INV3 0
using namespace std;

enum NET_TYPE {INPUT, INTERNAL, OUTPUT};
const double CAP1_SLEWS_CPU[] = {3.73582, 7.47165, 14.9433, 29.8866, 59.7732, 119.546, 239.093};
const double CAP2_SLEWS_CPU[] = {6.26418, 12.5284, 25.0567, 50.1134, 100.227, 200.454, 400.907};
__device__ const double CAP1_SLEWS[] = {3.73582, 7.47165, 14.9433, 29.8866, 59.7732, 119.546, 239.093};
__device__ const double CAP2_SLEWS[] = {6.26418, 12.5284, 25.0567, 50.1134, 100.227, 200.454, 400.907};
__managed__ int max_port_num;
__managed__ double *GOLDEN, *GOLDEN_INPUT;

#define idx(x, y, n) ((x) * (n) + (y))
#define idx3D(x, y, z, leny, lenz) (((x) * (leny) + (y)) * (lenz) + (z))

__managed__ int *sorted_stage_id;

int *net_type_cpu, *input_nets_cpu;
__managed__ int *net_type, *input_nets;//0: input net; 1: internal net: 2: output net

int *stage_node2_acc_num_cpu, *node2_acc_num_to_stage_cpu;
__managed__ int *stage_node2_acc_num, *node2_acc_num_to_stage;

int *stage_port2_acc_num_cpu, *port2_acc_num_to_stage_cpu;
__managed__ int *stage_port2_acc_num, *port2_acc_num_to_stage;

int *stage_portinode_acc_num_cpu, *portinode_acc_num_to_stage_cpu;
__managed__ int *stage_portinode_acc_num, *portinode_acc_num_to_stage;

int *stage_inode2_acc_num_cpu, *inode2_acc_num_to_stage_cpu;
__managed__ int *stage_inode2_acc_num, *inode2_acc_num_to_stage;

int *stage_port_acc_num_cpu;
__managed__ int *stage_port_acc_num;

int *RC_node_acc_num_cpu, *RC_inode2_acc_num_cpu, *RC_node2_acc_num_cpu, *RC_portinode_acc_num_cpu, *RC_input_node2_acc_num_cpu;
__managed__ int *RC_node_acc_num, *RC_inode2_acc_num, *RC_node2_acc_num, *RC_portinode_acc_num, *RC_input_node2_acc_num;

__managed__ double *net_delay, *stage_delay, *RC_port_delay;

double *RC_ccs_cap_cpu;
__managed__ double *RC_ccs_cap;

__managed__ double *stage_port_cap;

__managed__ double *input_g, *input_inv, *input_cap_g;

__managed__ double *GOLDEN_stage_port_cap;

__managed__ double *gate_ipin_delay, *output_delay;


int *RC_port_acc_num_cpu, *RC_port_gate_id_cpu, *RC_port_cell_id_cpu, *RC_port_pin_id_cpu, *RC_res_node0_cpu, *RC_res_node1_cpu, *cell_ipin_acc_num_cpu, *gate_ipin_acc_num_cpu;
double *RC_node_cap_cpu, *RC_res_cpu, *cell_ipin_nldm_cap_cpu;

int *ccs_acc_ilen_cpu, *ccs_acc_olen_cpu, *ccs_acc_tlen_cpu, *arc_acc_num_cpu;
double *ccs_refer_time_cpu, *ccs_input_slew_cpu, *ccs_output_cap_cpu, *ccs_tp_cpu;
int *stage_driver_gate_id_cpu, *stage_driver_ipin_cpu, *stage_signal_cpu, *stage_net_id_cpu, *stage_arc_id_cpu;


__managed__ int net_num, cell_num, gate_num, stage_num;
__managed__ double *RC_g, *RC_inv, *RC_cap_g;
__managed__ int *inode2_acc_num_to_net, *RC_port_acc_num, *RC_port_gate_id, *RC_port_cell_id, *RC_port_pin_id, *RC_res_node0, *RC_res_node1;
__managed__ int *cell_ipin_acc_num, *gate_ipin_acc_num, *stage_node_acc_num;
__managed__ double *RC_node_cap, *RC_res, *cell_ipin_nldm_cap, *basic_mat, *inslew, *stage_cap_g;
int *inode2_acc_num_to_net_cpu, *stage_node_acc_num_cpu;

__global__ void print_mat(double *mat, int n) {
    printf("print mat\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) printf("%13.3f", mat[idx(i, j, n)]);
        printf("\n");
    }    
}

struct net {

    net(string _name, NET_TYPE _net_type);

    void build_mx();

    int n;
    double total_cap;
    string net_name;
    NET_TYPE net_type;
    vector<pair<int, int>> gates;// (gate_id, pin_id); the first pair is the input of the net
    vector<double> caps;
    vector<tuple<int, int, double>> ress;
    mat g;
};

struct gate {

    gate(int _cell_id, int _gate_id, string _gate_name);
    void update_slew(double slew, int pin_id, int signal);
    void update_net_delay(double delay, int pin_id, int signal);
    void update_arc_delay(double delay, int arc_id, int signal);
    void update_input_at(double at, int pin_id, int signal);
    void update_output_at();

    vector<double> slews[2], i_ats[2], net_delays[2];// of input pins (0 for fall and 1 for rise)
    vector<double> o_ats[2];// of output pins
    vector<double> arc_delays[4];// of arcs (0/1/2/3 for fall->fall / fall->rise / rise->fall / fall->rise)

    string gate_name;
    int cell_id, gate_id;
    vector<int> i_nets, o_nets;
};

struct compute_arc {

    compute_arc(int _ipin, int _opin, int _gate_id, int _net_id, int _signal, const CCS::arc &arc);

    int ipin, opin, signal, gate_id, net_id;
    vector<double> delays;
    vector<double> slews;
    CCS::table2D nldm_slew, ccs_cap[2];
    CCS::ccs_table table;
};

struct design {

    //design(string name, string verilog_path, string spef_path);
    design(string name, string path);

    vector<vector<int>> levelize();
    void update_timing(double primary_input_slew, bool useCPU = false);
    void compute_delay(compute_arc &arc, int pass);
    void compute_delay_input(double input_slew, net &net, int signal);
    double compute_delay_internal(double input_slew, net &net, CCS::ccs_table &table, CCS::table2D &nldm, int signal, int isignal, int ipin, int driver_gate_id);
    void read_verilog(string filename);
    void read_spef(string filename);
    void add_pin_net(string pin, int net_id, int gate_id);
    void reduce_RC();
    void init_database_GPU();
    void init_graph_GPU();
    void prebuild();
    void read_sp_results();
    void read_pt_results();
    void prepare();

    vector<vector<int>> topo;
    string design_name, input_design_name, design_path;
    vector<net> nets;
    vector<gate> gates;
    map<string, double> po_net_delay[2], po_at[2];
    vector<int> input, output, level_stage_acc_num;
    vector<vector<int>> size_offset, size_max_node_num, size_num;
    vector<int> size_count;
    int max_RC_node_count, max_RC_port_count, max_RC_inode_count;
    int input_num, output_num;
};

void gate::update_slew(double slew, int pin_id, int signal) {
    slews[signal][pin_id] = max(slews[signal][pin_id], slew);
}
void gate::update_net_delay(double delay, int pin_id, int signal) {
    net_delays[signal][pin_id] = max(net_delays[signal][pin_id], delay);
}
void gate::update_arc_delay(double delay, int arc_id, int signal) {
    arc_delays[signal][arc_id] = max(arc_delays[signal][arc_id], delay);
}
void gate::update_input_at(double at, int pin_id, int signal) {
    i_ats[signal][pin_id] = max(i_ats[signal][pin_id], at);
}
void gate::update_output_at() {
    auto &cell = CCS::cells[cell_id];
    for(int ipin = 0; ipin < cell.get_ipin_size(); ipin++)
        for(int opin = 0; opin < cell.get_opin_size(); opin++) {
            int arc_id = cell.get_arc_index(ipin, opin);
            for(int s = 0; s < 4; s++) if(arc_delays[s][arc_id] != -1)
                o_ats[s % 2][opin] = max(o_ats[s % 2][opin], i_ats[s / 2][ipin] + arc_delays[s][arc_id]);
        }
}
map<tuple<int, int, int>, int> arc2idx_cell;// (gate_id, ipin, opin)->index
map<pair<int, int>, int> arc2index_net;// (net_id, port_id)->index
vector<double> sp_cell_fall, sp_cell_rise, sp_net_fall, sp_net_rise;
vector<double> pt_cell_fall, pt_cell_rise, pt_net_fall, pt_net_rise;
vector<double> my_cell_fall, my_cell_rise, my_net_fall, my_net_rise;
double *inslew_cpu;


set<string> cat[7];// XOR/XNOR, AND/NAND, OR/NOR, BUF/INV, AO/AOI, OA/OAI, MAJ
vector<int> cat_net[7], cat_cell[7];
int find_cell_type(string cell_name) {
    for(int i = 0; i < 7; i++) if(cat[i].find(cell_name) != cat[i].end()) return i;
    assert(0);
}
void design::prepare() {
    set<string> cell_types;
    for(int i = 0; i < gates.size(); i++) {
        auto &gate = gates[i];
        auto &cell = CCS::cells[gate.cell_id];
        cell_types.insert(cell.cell_name);
    }
    for(auto e : cell_types) {
        if(e.find("XOR") != string::npos || e.find("XNOR") != string::npos) cat[0].insert(e);
        else if(e.find("AND") != string::npos) cat[1].insert(e);
        else if(e.find("OR") != string::npos) cat[2].insert(e);
        else if(e.find("INV") != string::npos || e.find("BUF") != string::npos || e.find("HB1") != string::npos) cat[3].insert(e);
        else if(e.find("AO") != string::npos || e.find("A2O1A1") != string::npos) cat[4].insert(e);
        else if(e.find("OA") != string::npos || e.find("O2A1O1") != string::npos) cat[5].insert(e);
        else if(e.find("MAJ") != string::npos) cat[6].insert(e);
        else 
            cerr << e << " no category " << endl;
    }

    int cnt = 0;
    for(int i = 0; i < gates.size(); i++) {
        auto &gate = gates[i];
        auto &cell = CCS::cells[gate.cell_id];
        for(int ipin = 0; ipin < cell.i_pin_name.size(); ipin++)
            for(int opin = 0; opin < cell.o_pin_name.size(); opin++)
                arc2idx_cell[make_tuple(i, ipin, opin)] = cnt++;
    }
    assert(cnt == arc2idx_cell.size());
    cnt = 0;
    for(int i = 0; i < net_num; i++) if(net_type_cpu[i] == 1) {
        for(int j = 1; j < nets[i].gates.size(); j++)
            arc2index_net[make_pair(i, j)] = cnt++;
    }
    assert(cnt == arc2index_net.size());
}

void design::read_pt_results() {
    string filename = design_path +  "/pt_cell.results";
    FILE *file = fopen(filename.c_str(), "r");
    for(int i = 0; i < gates.size(); i++) {
        auto &gate = gates[i];
        auto &cell = CCS::cells[gate.cell_id];
        for(int ipin = 0; ipin < cell.i_pin_name.size(); ipin++)
            for(int opin = 0; opin < cell.o_pin_name.size(); opin++) {
                string from = gate.gate_name + "/" + cell.i_pin_name[ipin];
                string to = gate.gate_name + "/" + cell.o_pin_name[opin];
                double slew0, slew1, delay0, delay1;
                assert(4 == fscanf(file, "%lf%lf%lf%lf", &slew0, &slew1, &delay0, &delay1));
                inslew_cpu[gate_ipin_acc_num_cpu[i] + ipin] = slew0;
                inslew_cpu[gate_ipin_acc_num_cpu[gate_num] + gate_ipin_acc_num_cpu[i] + ipin] = slew1;
                cat_cell[find_cell_type(cell.cell_name)].emplace_back(pt_cell_fall.size());
                pt_cell_fall.emplace_back(delay0);
                pt_cell_rise.emplace_back(delay1);
            }
    }
    fclose(file);
    cudaMemcpy(inslew, inslew_cpu, sizeof(double) * gate_ipin_acc_num_cpu[gate_num] * 2, cudaMemcpyHostToDevice);
    filename = design_path +  "/pt_net.results";
    FILE *file2 = fopen(filename.c_str(), "r");
    for(int i = 0; i < net_num; i++) if(net_type_cpu[i] == 1) {
        auto &gate = gates[nets[i].gates[0].first];
        auto &cell = CCS::cells[gate.cell_id];
        for(int j = 1; j < nets[i].gates.size(); j++) {
            double d0, d1;
            assert(2 == fscanf(file2, "%lf%lf", &d0, &d1));
            cat_net[find_cell_type(cell.cell_name)].emplace_back(pt_net_fall.size());
            pt_net_fall.emplace_back(d0);
            pt_net_rise.emplace_back(d1);
        }
    }
    fclose(file2);
}
void design::read_sp_results() {
    string filename = design_path + "/spice_deck_all.txt";
    FILE *file = fopen(filename.c_str(), "r");
    sp_net_fall = sp_net_rise = vector<double> (arc2index_net.size(), 0);
    for(int i = 0; i < gates.size(); i++) {
        auto &gate = gates[i];
        auto &cell = CCS::cells[gate.cell_id];
        for(int ipin = 0; ipin < cell.i_pin_name.size(); ipin++)
            for(int opin = 0; opin < cell.o_pin_name.size(); opin++) {
                auto &net = nets[gate.o_nets[opin]];
                double val;
                for(int j = 0; j < net.gates.size(); j++) {
                    assert(1 == fscanf(file, "%lf", &val));
                    if(j == 0) sp_cell_fall.emplace_back(val);
                    else {
                        int net_id = gate.o_nets[opin];
                        if(net_type_cpu[net_id] != 1) continue;
                        int index = arc2index_net[make_pair(net_id, j)];
                        sp_net_fall[index] = max(sp_net_fall[index], val);
                    }
                }
                for(int j = 0; j < net.gates.size(); j++) {
                    assert(1 == fscanf(file, "%lf", &val));
                    if(j == 0) sp_cell_rise.emplace_back(val);
                    else {
                        int net_id = gate.o_nets[opin];
                        if(net_type_cpu[net_id] != 1) continue;
                        int index = arc2index_net[make_pair(net_id, j)];
                        sp_net_rise[index] = max(sp_net_rise[index], val);
                    }
                }
            }
    }
    fclose(file);
}

net::net(string _name, NET_TYPE _net_type) : net_name(_name), net_type(_net_type) {
    n = 0;
    gates.resize(1);
}

void net::build_mx() {
    g = mat(n, n);
    for(auto e : ress) g.addG(get<0> (e), get<1> (e), 1.0 / get<2> (e));
}

gate::gate(int _cell_id, int _gate_id, string _gate_name) : cell_id(_cell_id), gate_id(_gate_id), gate_name(_gate_name) {
    int ipin_size = CCS::cells[cell_id].get_ipin_size(), opin_size = CCS::cells[cell_id].get_opin_size();
    i_nets = vector<int> (ipin_size, -1);
    o_nets = vector<int> (opin_size, -1);
    for(int i = 0; i < 4; i++) {
        if(i < 2) {
            slews[i].resize(ipin_size);
            i_ats[i].resize(ipin_size);
            o_ats[i].resize(opin_size);
            net_delays[i].resize(ipin_size);
        }
        arc_delays[i] = vector<double> (ipin_size * opin_size, -1);
    }
}

design::design(string name, string path) {
    input_design_name = name;
    design_path = path;
    read_verilog(path + "/test.v");
    read_spef(path + "/test.spef");
    for(auto &net : nets) {
        net.total_cap = 0;
        for(auto cap : net.caps) net.total_cap += cap;
        for(int i = 1; i < net.gates.size(); i++) {
            auto &cell = CCS::cells[gates[net.gates[i].first].cell_id];
            net.total_cap += max(cell.nldm_cap[0][net.gates[i].second], cell.nldm_cap[1][net.gates[i].second]);
        }
    }
}

void design::add_pin_net(string pin, int net_id, int gate_id) {
    int ipin_id = CCS::cells[gates[gate_id].cell_id].get_ipin_index(pin);
    int opin_id = CCS::cells[gates[gate_id].cell_id].get_opin_index(pin);
    if(ipin_id == -1 && opin_id == -1) {
        cerr << "add_pin_net\n";
        exit(-1);
    }
    if(ipin_id != -1) {
        assert(nets[net_id].gates.size() > 0);
        nets[net_id].gates.emplace_back(gate_id, ipin_id);//needs to make sure the first one is the input of the net (output of a gate)
        gates[gate_id].i_nets[ipin_id] = net_id;
    } else {
        nets[net_id].gates[0] = make_pair(gate_id, opin_id);
        gates[gate_id].o_nets[opin_id] = net_id;
    }
    
}


void design::read_spef(string filename) {
    double _t = clock();
    auto tokens = tokenize(filename, "", "");

    const double RES_SCALE = tokens[find(tokens, "*R_UNIT", 0) + 2] == "OHM" ? 0.001 : 1;//scale resistance to KOHM
    const double CAP_SCALE = tokens[find(tokens, "*C_UNIT", 0) + 2] == "PF" ? 1000 : 1;//scale capacitance to FF
    

    //cout << "   RES_SCALE = " << setprecision(5) << RES_SCALE << endl;
    //cout << "   CAP_SCALE = " << setprecision(5) << CAP_SCALE << endl;


    map<string, int> net_name2index;
    for(int net_id = 0; net_id < nets.size(); net_id++) net_name2index[nets[net_id].net_name] = net_id;
    
    map<string, string> name_map;
    int cur = 0;
    while(tokens[cur] != "*D_NET" && tokens[cur] != "*NAME_MAP" && tokens[cur] != "*PORTS") cur++;
    if(tokens[cur] == "*NAME_MAP") {
        cur++;
        while(tokens[cur] != "*PORTS" && tokens[cur] != "*D_NET") {
            name_map[tokens[cur]] = tokens[cur + 1];
            cur += 2;
        }
    }

    auto get_name = [&name_map] (string name) {
        if(name[0] != '*') return name;
        int p = name.find(":");
        if(p == string::npos) {
            assert(name_map.count(name));
            return name_map[name];
        }
        assert(name_map.count(name.substr(0, p)));
        return name_map[name.substr(0, p)] +  name.substr(p);
    };

    for(cur = find(tokens, "*D_NET", cur); cur < tokens.size(); cur = find(tokens, "*D_NET", cur)) {
        assert(net_name2index.count(get_name(tokens[cur + 1])) > 0);
        int net_id = net_name2index[get_name(tokens[++cur])];
        auto &net = nets[net_id];

        
        map<string, int> node_name2index;
        for(int i = 0; i < net.gates.size(); i++) if(i != 0 || net.net_type != INPUT) {
            int gate_id = net.gates[i].first, pin_id = net.gates[i].second;
            auto pin_name = i == 0 ? CCS::cells[gates[gate_id].cell_id].o_pin_name[pin_id] :
                                        CCS::cells[gates[gate_id].cell_id].i_pin_name[pin_id];
            node_name2index[gates[gate_id].gate_name + ":" + pin_name] = net.n++;
        } else
            node_name2index[net.net_name] = net.n++;
        if(net.net_type == OUTPUT) {
            node_name2index[get_name(tokens[cur])] = net.n++;
        }
        net.caps.resize(net.n);
        cur = find(tokens, "*CAP", cur) + 1;
        while(all_of(tokens[cur].begin(), tokens[cur].end(), ::isdigit)) {
            auto node = get_name(tokens[++cur]);
            if(node_name2index.count(node) == 0) 
                node_name2index[node] = net.n++, net.caps.resize(net.n);
            net.caps[node_name2index[node]] = stod(tokens[++cur]) * CAP_SCALE;
            cur++;
        }     
        cur = find(tokens, "*RES", cur) + 1;
        while(all_of(tokens[cur].begin(), tokens[cur].end(), ::isdigit)) {
            string node1 = get_name(tokens[++cur]), node2 = get_name(tokens[++cur]);
            if(!node_name2index.count(node1)) {
                node_name2index[node1] = net.n++;
            }
            if(!node_name2index.count(node2)) {
                node_name2index[node2] = net.n++;
            }
            if(node_name2index.count(node1) && node_name2index.count(node2)) {
                net.ress.emplace_back(make_tuple(node_name2index[node1], node_name2index[node2], stod(tokens[++cur]) * RES_SCALE));
            } else {
                cur++;
                //cout << "Error: RC node without grounded cap " << node1 << ' ' << node2 << endl;
            }
            cur++;
        }
        net.caps.resize(net.n);

        assert(net.ress.size() + 1 >= node_name2index.size());//check connectivity

        for(auto e : node_name2index) assert(e.first[0] != '*');
        for(auto &e : net.caps) e = max(e, 1e-6);
    }
    for(auto e : net_name2index) assert(e.first[0] != '*');
}

void design::read_verilog(string filename) {
    auto tokens = tokenize(filename, "(),:;/#{}*\"", "().;");

    int cur = find(tokens, "module", 0);
    design_name = tokens[++cur];
    cur = find(tokens, ";", cur) + 1;

    map<string, int> name2index;
    int index_cnt = 0;

    for( ; cur < tokens.size(); cur++) {
        if(tokens[cur] == "endmodule") break;
        if(tokens[cur] == "input") {
            while(++cur < tokens.size() && tokens[cur] != ";")
                if(name2index.count(tokens[cur]) == 0) {
                    name2index[tokens[cur]] = index_cnt++;
                    input.emplace_back(nets.size());
                    nets.emplace_back(net(tokens[cur], INPUT));
                } else {
                    cerr << tokens[cur] << " Error Input\n";
                }
        } else if(tokens[cur] == "output") {
            while(++cur < tokens.size() && tokens[cur] != ";")
                if(name2index.count(tokens[cur]) == 0) {
                    name2index[tokens[cur]] = index_cnt++;
                    output.emplace_back(nets.size());
                    nets.emplace_back(net(tokens[cur], OUTPUT));
                } else
                    cerr << tokens[cur] << " Error Output\n";
        } else if(tokens[cur] == "wire") {
            while(++cur < tokens.size() && tokens[cur] != ";")
                if(name2index.count(tokens[cur]) == 0) {
                    name2index[tokens[cur]] = index_cnt++;
                    nets.emplace_back(net(tokens[cur], INTERNAL));
                }
        } else {
            gates.emplace_back(gate(CCS::get_cell_id(tokens[cur]), gates.size(), tokens[cur + 1]));
            auto &g = gates.back();
            cur += 2;
            string pin;
            for(; tokens[cur] != ";"; cur++) {
                if(tokens[cur][0] == '.')
                    pin = tokens[cur].substr(1);
                else if(tokens[cur] != "(" && tokens[cur] != ")") {
                    if(name2index.count(tokens[cur]) == 0) cerr << tokens[cur] << endl;
                    assert(name2index.count(tokens[cur]) > 0);
                    add_pin_net(pin, name2index[tokens[cur]], gates.size() - 1);
                }
            }
        }
    }
}


vector<vector<int>> design::levelize() {
    vector<vector<int>> topo_order(1);
    int cnt = gates.size();
    vector<int> prerequisite(gates.size(), 0);
    for(int gate_id = 0; gate_id < gates.size(); gate_id++) {
        for(auto net_id : gates[gate_id].i_nets) prerequisite[gate_id] += (nets[net_id].net_type != INPUT);
        if(prerequisite[gate_id] == 0) topo_order.back().emplace_back(gate_id), cnt--;
    }
    while(cnt > 0) {
        auto current_level = topo_order.back();
        topo_order.emplace_back(vector<int> ());
        for(auto gate_id : current_level) {
            for(auto net_id : gates[gate_id].o_nets) {
                for(int i = 1; i < nets[net_id].gates.size(); i++) {
                    int next_gate_id = nets[net_id].gates[i].first;
                    if(--prerequisite[next_gate_id] == 0) topo_order.back().emplace_back(next_gate_id), cnt--;
                }
            }
        }
    }
    return move(topo_order);
}




compute_arc::compute_arc(int _ipin, int _opin, int _gate_id, int _net_id, int _signal, const CCS::arc &arc) {
    ipin = _ipin;
    opin = _opin;
    signal = _signal;
    gate_id = _gate_id;
    net_id = _net_id;
    nldm_slew = arc.nldm_slew[signal & 1];
    ccs_cap[0] = arc.ccs_cap[signal & 1][0];
    ccs_cap[1] = arc.ccs_cap[signal & 1][1];
    table = arc.current[signal & 1];
}

const int CURRENT_SOURCE_ITERATION_NUMBER = 2;
const double STEP_NUM_INPUT = 10;
const double STEP_NUM[2] = {15, 20};
const double ESTIMATED_STEP_NUMBER = 20;
bool DEBUG = false;
double sim_time, inv_calc_time, inv_calc_time_iter0, sta_time;

void design::compute_delay(compute_arc &arc, int pass) {
    gate &gate = gates[arc.gate_id];
    net &net = nets[arc.net_id];

    assert(pass == 0 || pass == 1);
    double deltaT = (pass ? arc.delays[0] : arc.nldm_slew.lookup(gate.slews[arc.signal / 2][arc.ipin], net.total_cap)) / STEP_NUM[pass];
    double refer_time = arc.table.build(gate.slews[arc.signal / 2][arc.ipin]);
    

    int n = net.n, m = net.gates.size();// the number of nodes in the circuit
    vector<double> sim_times;
    vector<vector<double>> sim_voltages;

    auto get_time_vec = [&] (double percent) {
        vector<double> ans(n);
        for(int i = 0; i < n; i++) {
            vector<double> node_voltages(sim_times.size());
            for(int j = 0; j < node_voltages.size(); j++) node_voltages[j] = sim_voltages[j][i];
            ans[i] = interpolate1D(sim_times, get_index_coeff(node_voltages, percent * double(VDD)));
        }
        return ans;
    };

    vector<double> zero_vec(n, 0), cap_g, lastI, curI, lastV, curV;//conductance of grounded capacitors
    mat invG(n, n);

    auto compute_inv_g = [&] (int cap_type) {// compute the inverse conductance matrix
        cap_g = net.caps;
        for(int i = 1; i < net.gates.size(); i++) {
            auto &rec_gate = gates[net.gates[i].first];
            auto &rec_cell = CCS::cells[rec_gate.cell_id];
            double cap = 0;
            if(pass > 0) {
                for(int opin = 0; opin < rec_cell.o_pin_name.size(); opin++)
                    for(auto &rec_arc : rec_cell.arcs[rec_cell.get_arc_index(net.gates[i].second, opin)])
                        cap = max(cap, rec_arc.ccs_cap[arc.signal & 1][cap_type].lookup(arc.slews[i], nets[rec_gate.o_nets[opin]].total_cap));
            } else
                cap = CCS::cells[rec_gate.cell_id].nldm_cap[arc.signal & 1][net.gates[i].second];
            cap_g[i] += cap;
        }
        mat G = net.g;
        for(int i = 0; i < n; i++) G.addG(i, cap_g[i] = cap_g[i] * 2 / deltaT);
        
        double inv_calc_start_time = clock();
        invG = G.inv();
        inv_calc_time += clock() - inv_calc_start_time;
    };

    auto getV = [&] (double v, double t) {
        auto I_src = curV = curI = zero_vec;
        for(int i = 0; i < n; i++) I_src[i] = cap_g[i] * lastV[i] + lastI[i];
        I_src[0] += arc.table.getI(v, t);
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) curV[a] += invG.val[a][b] * I_src[b];
        for(int i = 0; i < n; i++) curI[i] = cap_g[i] * (curV[i] - lastV[i]) - lastI[i];
        return curV[0];
    };


    lastI = zero_vec;
    sim_times = {double(arc.table.tp[1][0][0])};
    sim_voltages = {lastV = vector<double> (n, 1) * double(deltaV)};

    double sim_start_time = clock();

    int cap_type = (pass == 0);
    compute_inv_g(cap_type);
    while(*min_element(sim_voltages.back().begin(), sim_voltages.back().end()) < 0.9 * VDD) {
        int cnt = 0;
        for(int i = 1; i < m; i++) cnt += (sim_voltages.back()[i] > 0.5 * VDD);
        if(sim_voltages.back()[1] > 0.5 * VDD && cap_type == 0) compute_inv_g(cap_type = 1);
        double v_src = sim_voltages.back()[0], t = sim_times.back() + deltaT;
        for(int _ = 0; _ < CURRENT_SOURCE_ITERATION_NUMBER; _++) v_src = getV(v_src, t);
        lastI = curI;
        sim_times.emplace_back(t);
        sim_voltages.emplace_back(lastV = curV);
    }

    sim_time += clock() - sim_start_time;
    

    auto time10 = get_time_vec(0.1), time50 = get_time_vec(0.5), time90 = get_time_vec(0.9);
    if(pass == 0) {
        arc.delays.resize(m);
        arc.slews.resize(m);
        arc.delays[0] = sim_times.back() - sim_times[0];
        for(int i = 0; i < m; i++) arc.slews[i] = time90[i] - time10[i];
    } else {
        for(int i = 0; i < m; i++) arc.delays[i] = time50[i] - (i == 0 ? refer_time : time50[0]);
        if(net.net_type == OUTPUT) po_net_delay[arc.signal & 1][net.net_name] = time50[1] - time50[0];
    }
}

int DEBUG_ID;
double MAX_CAP_DIFF = 0;
int MAX_CAP_CNT = 0;

double design::compute_delay_internal(double input_slew, net &net, CCS::ccs_table &table, CCS::table2D &nldm, int signal, int isignal, int ipin, int driver_gate_id) {

    double deltaT = DELTA_T;


    double update_start_time = clock();
    auto refer_time = table.build(input_slew);

    int n = net.n;// the number of nodes in the circuit
    vector<double> sim_times;
    vector<vector<double>> sim_voltages;

    auto get_time_vec = [&] (double percent) {
        vector<double> ans(n);
        for(int i = 0; i < n; i++) {
            vector<double> node_voltages(sim_times.size());
            for(int j = 0; j < node_voltages.size(); j++) node_voltages[j] = sim_voltages[j][i];
            ans[i] = interpolate1D(sim_times, get_index_coeff(node_voltages, percent * double(VDD)));
        }
        return ans;
    };


    vector<double> zero_vec(n, 0), cap_g, lastI, curI, lastV, curV;//conductance of grounded capacitors
    mat invG(n, n);

    auto compute_inv_g = [&] (vector<double> slews, int iter, bool nldm_cap) {// compute the inverse conductance matrix
        double start_time = clock();
        cap_g = net.caps;
        for(int i = 1; i < net.gates.size(); i++) {
            const auto &gate = gates[net.gates[i].first];
            auto &cell = CCS::cells[gate.cell_id];
            if(!nldm_cap) {
                double cap = 0;
                double min_cap = 1000;
                int cnt = 0;
                for(int opin_id = 0; opin_id < cell.get_opin_size(); opin_id++) {
                    
                    int arc_idx = cell.get_arc_index(net.gates[i].second, opin_id);
                    for(auto &arc : cell.arcs[arc_idx]) {
                        double new_cap = arc.ccs_cap[signal][iter].lookup(slews[i], nets[gate.o_nets[opin_id]].total_cap);
                        cap = max(cap, new_cap);
                        min_cap = min(min_cap, new_cap);
                        cnt++;
                    }
                }
                MAX_CAP_DIFF = max(MAX_CAP_DIFF, cap - min_cap);
                MAX_CAP_CNT = max(MAX_CAP_CNT, cnt);
                //GOLDEN_stage_port_cap[(stage_port_acc_num_cpu[DEBUG_ID] + i) * 2 + iter] = cap;
                //if(cap - min_cap > 1) printf("min_cap=%.4f  cap=%.4f\n", min_cap, cap);
                cap_g[i] += cap;
            } else {
                cap_g[i] += cell.nldm_cap[signal][net.gates[i].second];
            }
        }

        for(auto &cap : cap_g) cap *= 2 / deltaT;
        mat G = net.g;

        for(int i = 0; i < n; i++) {
            G.addG(i, cap_g[i]);
        }
        double inv_calc_start_time = clock();
        invG = G.inv();
        inv_calc_time += clock() - inv_calc_start_time;
        if(nldm_cap) inv_calc_time_iter0 += clock() - inv_calc_start_time;
    };
    
    auto getV = [&] (double v, double t) {
        auto I_src = curV = curI = zero_vec;
        for(int i = 0; i < n; i++) I_src[i] = cap_g[i] * lastV[i] + lastI[i];
        I_src[0] += table.getI(v, t);
        for(int a = 0; a < n; a++) for(int b = 0; b < n; b++) curV[a] += invG.val[a][b] * I_src[b];
        for(int i = 0; i < n; i++) curI[i] = cap_g[i] * (curV[i] - lastV[i]) - lastI[i];
        return curV[0];
    };


    auto simulate = [&] (vector<double> slews1, vector<double> slews2, bool nldm_cap) {
        double _t = clock();
        lastI = zero_vec;
        sim_times = {double(table.tp[1][0][0])};
        sim_voltages = {lastV = vector<double> (n, 1) * double(deltaV)};

        int cap_type = 0;
        compute_inv_g(slews1, cap_type, nldm_cap);
        while(*min_element(sim_voltages.back().begin(), sim_voltages.back().end()) < 0.9 * VDD) {
            if(!nldm_cap && cap_type == 0 && sim_voltages.back()[1] >= 0.5 * VDD) {
                compute_inv_g(slews2, cap_type = 1, nldm_cap);
            }
            double v_src = sim_voltages.back()[0], t = sim_times.back() + deltaT;
            for(int _ = 0; _ < CURRENT_SOURCE_ITERATION_NUMBER; _++) v_src = getV(v_src, t);
            
            lastI = curI;
            sim_times.emplace_back(t);
            sim_voltages.emplace_back(lastV = curV);
        }        
    };


    
    double sim_start_time = clock();
    vector<double> slews, slews1, slews2;
    for(int _ = 0; _ < 2; _++) {
        simulate(slews1, slews2, _ == 0);
        if(_ == 1) continue;
        slews = slews1 = slews2 = get_time_vec(0.9) - get_time_vec(0.1);

        //deltaT = (sim_times.back() - sim_times[0]) / 20;
    }


    auto times_50 = get_time_vec(0.5);


    for(int i = 1; i < net.gates.size(); i++) {
        auto &gate = gates[net.gates[i].first];
        gate.update_slew(slews[i], net.gates[i].second, signal);
        //gate.update_input_at(gates[driver_gate_id].i_ats[isignal][ipin] + times_50[i] - refer_time, net.gates[i].second, signal);
        gate.update_net_delay(times_50[i] - times_50[0], net.gates[i].second, signal);
    }

    if(net.net_type == OUTPUT) po_net_delay[signal][net.net_name] = times_50[1] - times_50[0];
    //if(net.net_type == OUTPUT) po_at[signal][net.net_name] = gates[driver_gate_id].i_ats[isignal][ipin] + times_50[1] - refer_time;

    //GOLDEN[DEBUG_ID * max_RC_port_count] = times_50[0] - refer_time;
    //for(int i = 0; i < net.gates.size(); i++) GOLDEN[DEBUG_ID * max_RC_port_count + i] = times_50[i] - refer_time;

    return times_50[0] - refer_time;
}


//__device__ const double VDD = 0.7;


__device__ double lerp(double x0, double x1, double y0, double y1, double y) {
    return (y - y0) / (y1 - y0) * (x1 - x0) + x0;
}
__device__ double lerp(double x0, double x1, double coeff) {
    return x0 * (1 - coeff) + x1 * coeff;
}

__global__ void calc_delay_input_0_small(double input_slew, int signal) {
    extern __shared__ double shared[];//(n+1)*(2n+3)
    __shared__ int achieve_num;
    if(threadIdx.x == 0) achieve_num = 0;

    int net_id = input_nets[blockIdx.x];
    int node_offset = RC_node_acc_num[net_id - 1];
    int port_offset = RC_port_acc_num[net_id - 1];
    int n = RC_node_acc_num[net_id] - RC_node_acc_num[net_id - 1];
    int m = RC_port_acc_num[net_id] - RC_port_acc_num[net_id - 1];

    assert(blockDim.x > n);

    const double deltaT = input_slew / 10, V_T_SLOPE = 0.8 * VDD / input_slew;

    double cap_g, *I_src = shared, *g = I_src + n + 1, *inv = g + (n + 1) * (n + 1);

    //initialize g & inv
    for(int i = 0; i * blockDim.x + threadIdx.x < (n + 1) * (n + 1); i++) 
        g[i * blockDim.x + threadIdx.x] = inv[i * blockDim.x + threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < n) cap_g = RC_node_cap[node_offset + threadIdx.x];
    if(threadIdx.x <= n) inv[idx(threadIdx.x, threadIdx.x, n + 1)] = 1;
    __syncthreads();
    if(1 <= threadIdx.x && threadIdx.x < m) cap_g += cell_ipin_nldm_cap[cell_ipin_acc_num[cell_num] * signal
        + cell_ipin_acc_num[RC_port_cell_id[port_offset + threadIdx.x] - 1] + RC_port_pin_id[port_offset + threadIdx.x]];
    if(threadIdx.x < n)
        g[idx(threadIdx.x, threadIdx.x, n + 1)] += (cap_g *= 2 / deltaT);
    __syncthreads();
    if(threadIdx.x < n && RC_res[node_offset + threadIdx.x] > 0) {
        int u = RC_res_node0[node_offset + threadIdx.x], v = RC_res_node1[node_offset + threadIdx.x];
        double conductance = 1.0 / RC_res[node_offset + threadIdx.x];
        atomicAdd(g + idx(u, u, n + 1), conductance);
        atomicAdd(g + idx(v, v, n + 1), conductance);
        atomicAdd(g + idx(u, v, n + 1), -conductance);
        atomicAdd(g + idx(v, u, n + 1), -conductance);
    }
    if(threadIdx.x == 0) g[idx(n, 0, n + 1)] = g[idx(0, n, n + 1)] = 1;
    __syncthreads();

    //compute inv from g
    for(int i = 0; i <= n; i++)
        for(int j = i + 1; j <= n; j++) {
            double coeff = g[idx(j, i, n + 1)] / g[idx(i, i, n + 1)];
            __syncthreads();
            if(threadIdx.x <= n) {
                g[idx(j, threadIdx.x, n + 1)] -= coeff * g[idx(i, threadIdx.x, n + 1)];
                inv[idx(j, threadIdx.x, n + 1)] -= coeff * inv[idx(i, threadIdx.x, n + 1)];
            }
            __syncthreads();
        }
    for(int i = n; i >= 0; i--) {
        for(int j = 0; j < i; j++) {
            double coeff = g[idx(j, i, n + 1)] / g[idx(i, i, n + 1)];
            __syncthreads();
            if(threadIdx.x <= n) {
                g[idx(j, threadIdx.x, n + 1)] -= coeff * g[idx(i, threadIdx.x, n + 1)];
                inv[idx(j, threadIdx.x, n + 1)] -= coeff * inv[idx(i, threadIdx.x, n + 1)];
            }
            __syncthreads();
        }
        if(threadIdx.x <= n) inv[idx(i, threadIdx.x, n + 1)] /= g[idx(i, i, n + 1)];
    }
    __syncthreads();

    double lastI = 0, lastV = 0, curV = 0, sim_time = 0, port_slew = 0;
    while(1) {
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        if(threadIdx.x == n) I_src[threadIdx.x] = min(VDD, V_T_SLOPE * (sim_time + deltaT));
        __syncthreads();
        if(threadIdx.x < n) {
            curV = 0;
            for(int i = 0; i <= n; i++) curV += inv[idx(threadIdx.x, i, n + 1)] * I_src[i];
            lastI = cap_g * (curV - lastV) - lastI;
            if(threadIdx.x < m && lastV < 0.1 * VDD && curV >= 0.1 * VDD)
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.1 * VDD);
            if(threadIdx.x < m && lastV < 0.9 * VDD && curV >= 0.9 * VDD) {
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.9 * VDD) - port_slew;
                atomicAdd(&achieve_num, 1);
            }
            lastV = curV;
        }
        __syncthreads();
        sim_time += deltaT;
        if(achieve_num == m) break;
    }
    if(threadIdx.x < m)
        printf("test%2d slew[%d] = %.4f\n", blockIdx.x, threadIdx.x, port_slew);
}

__global__ void calc_delay_input_0_large(double input_slew, int signal, int net_offset) {
    extern __shared__ double I_src[];
    __shared__ int achieve_num;
    if(threadIdx.x == 0) achieve_num = 0;

    int net_id = input_nets[blockIdx.x + net_offset];
    int node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - node_offset;
    int port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - port_offset;
    
    double *g = input_g + RC_input_node2_acc_num[net_id];
    double *inv = input_inv + RC_input_node2_acc_num[net_id];

    assert(blockDim.x > n);

    const double deltaT = input_slew / 10, V_T_SLOPE = 0.8 * VDD / input_slew;

    double cap_g;

    

    //initialize g & inv
    for(int i = 0; i * blockDim.x + threadIdx.x < (n + 1) * (n + 1); i++) 
        g[i * blockDim.x + threadIdx.x] = inv[i * blockDim.x + threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < n) cap_g = RC_node_cap[node_offset + threadIdx.x];
    if(threadIdx.x <= n) inv[idx(threadIdx.x, threadIdx.x, n + 1)] = 1;
    __syncthreads();
    if(1 <= threadIdx.x && threadIdx.x < m) cap_g += cell_ipin_nldm_cap[cell_ipin_acc_num[cell_num] * signal +
        cell_ipin_acc_num[RC_port_cell_id[port_offset + threadIdx.x]] + RC_port_pin_id[port_offset + threadIdx.x]];
    if(threadIdx.x < n)
        g[idx(threadIdx.x, threadIdx.x, n + 1)] += (cap_g *= 2 / deltaT);
    __syncthreads();
    if(threadIdx.x < n && RC_res[node_offset + threadIdx.x] > 0) {
        int u = RC_res_node0[node_offset + threadIdx.x], v = RC_res_node1[node_offset + threadIdx.x];
        double conductance = 1.0 / RC_res[node_offset + threadIdx.x];
        atomicAdd(g + idx(u, u, n + 1), conductance);
        atomicAdd(g + idx(v, v, n + 1), conductance);
        atomicAdd(g + idx(u, v, n + 1), -conductance);
        atomicAdd(g + idx(v, u, n + 1), -conductance);
    }
    if(threadIdx.x == 0) g[idx(n, 0, n + 1)] = g[idx(0, n, n + 1)] = 1;
    __syncthreads();


    //compute inv from g
    for(int i = 0; i <= n; i++)
        for(int j = i + 1; j <= n; j++) {
            double coeff = g[idx(j, i, n + 1)] / g[idx(i, i, n + 1)];
            __syncthreads();
            if(threadIdx.x <= n) {
                g[idx(j, threadIdx.x, n + 1)] -= coeff * g[idx(i, threadIdx.x, n + 1)];
                inv[idx(j, threadIdx.x, n + 1)] -= coeff * inv[idx(i, threadIdx.x, n + 1)];
            }
            __syncthreads();
        }
    for(int i = n; i >= 0; i--) {
        for(int j = 0; j < i; j++) {
            double coeff = g[idx(j, i, n + 1)] / g[idx(i, i, n + 1)];
            __syncthreads();
            if(threadIdx.x <= n) {
                g[idx(j, threadIdx.x, n + 1)] -= coeff * g[idx(i, threadIdx.x, n + 1)];
                inv[idx(j, threadIdx.x, n + 1)] -= coeff * inv[idx(i, threadIdx.x, n + 1)];
            }
            __syncthreads();
        }
        if(threadIdx.x <= n) inv[idx(i, threadIdx.x, n + 1)] /= g[idx(i, i, n + 1)];
    }
    __syncthreads();
    double lastI = 0, lastV = 0, curV = 0, sim_time = 0, port_slew = 0;
    while(1) {
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        if(threadIdx.x == n) I_src[threadIdx.x] = min(VDD, V_T_SLOPE * (sim_time + deltaT));
        __syncthreads();
        if(threadIdx.x < n) {
            curV = 0;
            for(int i = 0; i <= n; i++) curV += inv[idx(threadIdx.x, i, n + 1)] * I_src[i];
            lastI = cap_g * (curV - lastV) - lastI;
            if(threadIdx.x < m && lastV < 0.1 * VDD && curV >= 0.1 * VDD)
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.1 * VDD);
            if(threadIdx.x < m && lastV < 0.9 * VDD && curV >= 0.9 * VDD) {
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.9 * VDD) - port_slew;
                atomicAdd(&achieve_num, 1);
            }
            lastV = curV;
        }
        __syncthreads();
        sim_time += deltaT;
        if(achieve_num == m) break;
    }    
    if(0 < threadIdx.x && threadIdx.x < m) inslew[gate_ipin_acc_num[gate_num] * signal + gate_ipin_acc_num[
        RC_port_gate_id[port_offset + threadIdx.x]] + RC_port_pin_id[port_offset + threadIdx.x]] = port_slew;
}

__global__ void calc_delay_input_new(int *input_node2_acc_num, int tot_node2, double input_slew, int signal) {
    extern __shared__ double I_src[];
    __shared__ int achieve_num;
    if(threadIdx.x == 0) achieve_num = 0;


    int net_id = input_nets[blockIdx.x], node2_offset = signal * tot_node2 + input_node2_acc_num[blockIdx.x];
    int node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - node_offset;
    int port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - port_offset;
    assert(blockDim.x > n);


    const double deltaT = DELTA_T, V_T_SLOPE = 0.8 * VDD / input_slew;
    double *inv = input_inv + node2_offset, cap_g = 0;
    if(threadIdx.x < n) cap_g = input_cap_g[node2_offset + threadIdx.x];


    double lastI = 0, lastV = 0, curV = 0, sim_time = 0, port_slew = 0;
    while(1) {
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        if(threadIdx.x == n) I_src[threadIdx.x] = min(VDD, V_T_SLOPE * (sim_time + deltaT));
        __syncthreads();
        if(threadIdx.x < n) {
            curV = 0;
            for(int i = 0; i <= n; i++) curV += inv[idx(threadIdx.x, i, n + 1)] * I_src[i];            
            lastI = cap_g * (curV - lastV) - lastI;
            if(threadIdx.x < m && lastV < 0.1 * VDD && curV >= 0.1 * VDD)
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.1 * VDD);
            if(threadIdx.x < m && lastV < 0.9 * VDD && curV >= 0.9 * VDD) {
                port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.9 * VDD) - port_slew;
                atomicAdd(&achieve_num, 1);
            }
            lastV = curV;
        }
        __syncthreads();
        sim_time += deltaT;
        if(achieve_num == m) break;
    }
    if(EVALUATE == 0 && 0 < threadIdx.x && threadIdx.x < m) inslew[gate_ipin_acc_num[gate_num] * signal + gate_ipin_acc_num[
        RC_port_gate_id[port_offset + threadIdx.x]] + RC_port_pin_id[port_offset + threadIdx.x]] = port_slew;
}



__managed__ int ccstable_num;
__device__ const int VPLEN = 14;
__device__ const double VP[] = {0.01 * VDD, 0.02 * VDD, 0.05 * VDD, 0.1 * VDD, 0.2 * VDD, 0.3 * VDD, 0.4 * VDD, 0.5 * VDD, 0.6 * VDD, 0.7 * VDD, 0.8 * VDD, 0.9 * VDD, 0.95 * VDD, 0.97 * VDD};
__managed__ int *stage_driver_gate_id, *stage_driver_ipin, *stage_signal, *stage_net_id, *stage_arc_id;
__managed__ int *ccs_acc_ilen, *ccs_acc_olen, *ccs_acc_tlen;
__managed__ double *ccs_refer_time, *ccs_input_slew, *ccs_output_cap, *ccs_tp, *stage_g, *stage_inv;

__device__ double inbound(double x, double l, double r) {
    if(x < l) return l;
    if(x > r) return r;
    return x;
}



__device__ double cuda_getI(double v, double t, int olen, double *tp, double *output_cap) {
    int v_idx = 0, t_idx = 0;
    double cap2time[10], v_coeff, t_coeff;
    v = inbound(v, VP[0], VP[VPLEN - 1]);
    for(int i = 1; i < VPLEN; i++) if(VP[i - 1] < v && v <= VP[i]) { v_idx = i - 1; break; }
    v_coeff = (v - VP[v_idx]) / (VP[v_idx + 1] - VP[v_idx]);
    for(int i = 0; i < olen; i++) cap2time[i] = lerp(tp[idx3D(1, v_idx, i, VPLEN, olen)], tp[idx3D(1, v_idx + 1, i, VPLEN, olen)], v_coeff);
    t = inbound(t, cap2time[0], cap2time[olen - 1]);
    for(int i = 1; i < olen; i++) if(cap2time[i - 1] < t && t <= cap2time[i]) { t_idx = i - 1; break; }
    t_coeff = (t - cap2time[t_idx]) / (cap2time[t_idx + 1] - cap2time[t_idx]);
    double deltaT0 = lerp(tp[idx3D(2, v_idx, t_idx, VPLEN, olen)], tp[idx3D(2, v_idx, t_idx + 1, VPLEN, olen)], t_coeff)
        - lerp(tp[idx3D(0, v_idx, t_idx, VPLEN, olen)], tp[idx3D(0, v_idx, t_idx + 1, VPLEN, olen)], t_coeff);
    double deltaT1 = lerp(tp[idx3D(2, v_idx + 1, t_idx, VPLEN, olen)], tp[idx3D(2, v_idx + 1, t_idx + 1, VPLEN, olen)], t_coeff)
        - lerp(tp[idx3D(0, v_idx + 1, t_idx, VPLEN, olen)], tp[idx3D(0, v_idx + 1, t_idx + 1, VPLEN, olen)], t_coeff);
    return lerp(output_cap[t_idx], output_cap[t_idx + 1], t_coeff) * 0.02 * VDD * ((1 - v_coeff) / deltaT0 + v_coeff / deltaT1);
    //return lerp(output_cap[t_idx], output_cap[t_idx + 1], t_coeff) * 0.02 * VDD * (1.0 / (1 - v_coeff) / deltaT0 + 1.0 / v_coeff / deltaT1);

}
__device__ void atomicMax64(double* address, double val)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        if(__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

void design::compute_delay_input(double input_slew, net &net, int signal) {
    double deltaT = input_slew / STEP_NUM_INPUT;// simulation time step: deltaT ps   
    const double V_T_SLOPE = 0.8 * VDD / input_slew;

    int n = net.n;
    vector<double> sim_times;
    vector<vector<double>> sim_voltages;


    auto get_time_vec = [&] (double percent) {
        vector<double> ans(n);
        for(int i = 0; i < n; i++) {
            vector<double> node_voltages(sim_times.size());
            for(int j = 0; j < node_voltages.size(); j++) node_voltages[j] = sim_voltages[j][i];
            ans[i] = interpolate1D(sim_times, get_index_coeff(node_voltages, percent * double(VDD)));
        }
        return ans;
    };


    vector<double> zero_vec(n, 0), cap_g, lastI, curI, lastV, curV;
    mat invG(n, n);

    auto compute_inv_g = [&] (vector<double> slews, int iter, bool nldm_cap) {
        cap_g = net.caps;
        for(int i = 1; i < net.gates.size(); i++) {
            const auto &gate = gates[net.gates[i].first];
            auto &cell = CCS::cells[gate.cell_id];
            if(!nldm_cap) {
                iter = sim_voltages.back()[i] > 0.5 * VDD ? 1 : 0;
                double cap = 0;
                for(int opin_id = 0; opin_id < cell.get_opin_size(); opin_id++) {
                    int arc_idx = cell.get_arc_index(net.gates[i].second, opin_id);
                    for(auto &arc : cell.arcs[arc_idx]) {
                        cap = max(cap, arc.ccs_cap[signal][iter].lookup(slews[i], nets[gate.o_nets[opin_id]].total_cap));
                    }
                }
                cap_g[i] += cap;
            } else {
                cap_g[i] += cell.nldm_cap[signal][net.gates[i].second];
            }
        }
        for(auto &cap : cap_g) cap *= 2 / deltaT;
        mat G = mat(n + 1, n + 1);
        for(auto e : net.ress) G.addG(get<0> (e), get<1> (e), 1.0 / get<2> (e));
        for(int i = 0; i < n; i++) G.addG(i, cap_g[i]);
        G.val[n][0] = G.val[0][n] = 1;
        double inv_calc_start_time = clock();
        invG = G.inv();
        inv_calc_time += clock() - inv_calc_start_time;
        if(nldm_cap) inv_calc_time_iter0 += clock() - inv_calc_start_time;
    };
    auto computeV = [&] (double v, double t) {
        auto I_src = curV = curI = zero_vec;
        for(int i = 0; i < n; i++) I_src[i] = cap_g[i] * lastV[i] + lastI[i];
        I_src.emplace_back(v);
        for(int a = 0; a < n; a++) for(int b = 0; b <= n; b++) curV[a] += invG.val[a][b] * I_src[b];
        for(int i = 0; i < n; i++) curI[i] = cap_g[i] * (curV[i] - lastV[i]) - lastI[i];
    };

    auto waveform = [&] (double t) {
        return min(VDD, V_T_SLOPE * t);
        //vector<double> times = {0, 1.5, 2.5, 3.375, 4.375, 5.375, 6.375, 7.375, 8.5, 10.75, 12, 13.25, 16, 19, 22.5, 24.875, 26.625};
        //vector<double> times = {0, 0.375, 0.625, 0.84375, 1.09375, 1.34375, 1.59375, 1.84375, 2.125, 2.6875, 3, 3.3125, 4, 4.75, 5.625, 6.21875, 6.65625};
        //vector<double> volts = vector<double> ({0, 0.03, 0.1, 0.158744, 0.221271, 0.279374, 0.333513, 0.3841, 0.437223, 0.533203, 0.58153, 0.626864, 0.717883, 0.806555, 0.9, 0.958983, 1}) * VDD;
        //return interpolate1D(volts, get_index_coeff(times, min(26.625, t)));
    };

    auto simulate = [&] (vector<double> slews1, vector<double> slews2, bool nldm_cap) {
        lastI = zero_vec;
        sim_times = {0};
        sim_voltages = {lastV = zero_vec};

        int cap_type = 0;
        compute_inv_g(slews1, cap_type, nldm_cap);
        while(*min_element(sim_voltages.back().begin(), sim_voltages.back().end()) < 0.9 * VDD) {      
            if(!nldm_cap && cap_type == 0 && sim_voltages.back()[1] >= 0.5 * VDD)
                compute_inv_g(slews2, cap_type = 1, nldm_cap);
            double t = sim_times.back() + deltaT, v_src = waveform(t);//V_T_SLOPE * t;
            computeV(v_src, t);
            lastI = curI;
            sim_times.emplace_back(t);
            sim_voltages.emplace_back(lastV = curV);
        }
    };
    

    double sim_start_time = clock();

    vector<double> slews, slews1, slews2;
    for(int _ = 0; _ < 2; _++) {
        simulate(slews1, slews2, _ == 0);
        auto time10 = get_time_vec(0.1), time50 = get_time_vec(0.5), time90 = get_time_vec(0.9);
        if(_ == 1) continue;
        slews = slews1 = slews2 = time90 - time10;
        deltaT = (sim_times.back() - sim_times[0]) / (STEP_NUM_INPUT + 5);
        //slews1 = (get_time_vec(0.5) - get_time_vec(0.1)) * 2;
        //slews2 = (get_time_vec(0.9) - get_time_vec(0.5)) * 2;
    }

    sim_time += clock() - sim_start_time;

    auto times_50 = get_time_vec(0.5);


    for(int i = 1; i < net.gates.size(); i++) {
        auto &gate = gates[net.gates[i].first];
        gate.update_slew(slews[i], net.gates[i].second, signal);
        gate.update_net_delay(times_50[i] - times_50[0], net.gates[i].second, signal);
        gate.update_input_at(times_50[i] - times_50[0], net.gates[i].second, signal);
    }
}



const double RC_REDUCTION_RATE = 0.017, RC_REDUCTION_THRESHOLD = 64;
void design::reduce_RC() {

    function<int(int, vector<int> &)> root = [&root] (int x, vector<int> &fa) {
        return x == fa[x] ? x : fa[x] = root(fa[x], fa);
    };

    double tot_reduction = 0;

    int max_node_count = 256;

    for(auto &net : nets) {
        if(net.n <= RC_REDUCTION_THRESHOLD) continue;
        vector<double> all_res;
        double max_res = 0;
        for(auto &e : net.ress) all_res.emplace_back(get<2> (e));
        for(auto &e : net.ress) max_res = max(max_res, get<2> (e));
        //sort(all_res.begin(), all_res.end(), greater<double> ());
        //double max_res2 = all_res.size() <= max_node_count ? 0 : all_res[max_node_count];
        max_res *= RC_REDUCTION_RATE * (1.0 * net.ress.size() / RC_REDUCTION_THRESHOLD);
        //max_res = max(max_res, max_res2);
        vector<int> fa(net.n);
        for(int i = 0; i < net.n; i++) fa[i] = i;
        for(auto &e : net.ress) if(get<2> (e) < max_res) {
            int a = root(get<0> (e), fa), b = root(get<1> (e), fa);
            if(max(a, b) < net.gates.size()) continue;
            fa[max(a, b)] = min(a, b);
        }
        for(int i = 0; i < net.gates.size(); i++) assert(root(i, fa) == i);
        int new_n = 0;
        vector<int> new_id(net.n, -1);
        for(int i = 0; i < net.n; i++) if(root(i, fa) == i) new_id[i] = new_n++;
        vector<double> new_caps(new_n, 0);
        for(int i = 0; i < net.n; i++) new_caps[new_id[root(i, fa)]] += net.caps[i];
        vector<tuple<int, int, double>> new_ress;
        for(auto e : net.ress) {
            int a = root(get<0> (e), fa), b = root(get<1> (e), fa);
            if(a != b) new_ress.emplace_back(make_tuple(new_id[a], new_id[b], get<2> (e)));
        }

        tot_reduction += 1 - 1.0 * new_n / net.n;


        net.n = new_n;
        net.ress = new_ress;
        net.caps = new_caps;
    }

    output_log("res/size thresholds: " + to_string(RC_REDUCTION_RATE) + "/" + to_string(RC_REDUCTION_THRESHOLD) + "; actual reduction: " + to_string(tot_reduction / nets.size()));
}


void print_GPU_mem() {
    size_t free_byte, total_byte ;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte, total_db = (double)total_byte;
    printf("GPU memory usage: used/total = %.1f/%.1f (GB)\n", (total_byte - free_byte)/1024/1024.0/1024, total_byte/1024/1024.0/1024);
}


__global__ void build_RC_g() {
    int net_id = blockIdx.x, RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    double *g = RC_g + RC_node2_acc_num[net_id], *cap_g = RC_cap_g + RC_node_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];

    for(int i = 0; i * blockDim.x + threadIdx.x < n * n; i++) {
        int idx = i * blockDim.x + threadIdx.x, x = idx / n, y = idx % n;
        g[idx] = 0;
        inv[idx] = (x == y ? 1 : 0);
    }
    __syncthreads();
    if(threadIdx.x < n) {
        if(RC_res[RC_node_offset + threadIdx.x] > 0) {
            int u = RC_res_node0[RC_node_offset + threadIdx.x], v = RC_res_node1[RC_node_offset + threadIdx.x];
            double conductance = 1.0 / RC_res[RC_node_offset + threadIdx.x];
            atomicAdd(g + idx(u, u, n), conductance);
            atomicAdd(g + idx(v, v, n), conductance);
            atomicAdd(g + idx(u, v, n), -conductance);
            atomicAdd(g + idx(v, u, n), -conductance);
        }
        cap_g[threadIdx.x] = RC_node_cap[RC_node_offset + threadIdx.x] * 2 / DELTA_T;
        atomicAdd(g + idx(threadIdx.x, threadIdx.x, n), cap_g[threadIdx.x]);
    }
}
__global__ void build_RC_invD(int tot_inode2, int pivot, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot_inode2) return;
    int net_id = inode2_acc_num_to_net[idx];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    pivot += m;
    if(pivot >= n) return;
    int x = (idx - RC_inode2_acc_num[net_id]) / (n - m) + m, y = (idx - RC_inode2_acc_num[net_id]) % (n - m) + m;
    assert(x < n);
    double *g = RC_g + RC_node2_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];
    if(type == 0) {
        if(x > pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        }
    } else if(type == 1) {
        if(x < pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        } 
    } else if(type == 2) {
        inv[idx(x, y, n)] /= g[idx(x, x, n)];
    }    
}
/*
__global__ void sorted_build_RC_invD(int tot_inode2, int pivot, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot_inode2) return;
    int net_id = sorted_net_id[sorted_inode2_acc_num_to_net[idx]];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    pivot += m;
    if(pivot >= n) return;
    int x = (idx - RC_inode2_acc_num[net_id]) / (n - m) + m, y = (idx - RC_inode2_acc_num[net_id]) % (n - m) + m;
    assert(x < n);
    double *g = RC_g + RC_node2_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];
    if(type == 0) {
        if(x > pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        }
    } else if(type == 1) {
        if(x < pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        } 
    } else if(type == 2) {
        inv[idx(x, y, n)] /= g[idx(x, x, n)];
    }    
}*/

__global__ void build_RC_invBC() {
    int net_id = blockIdx.x;
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    double *g = RC_g + RC_node2_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < n * n; i++) {
        int x = (i * blockDim.x + threadIdx.x) / n, y = (i * blockDim.x + threadIdx.x) % n;
        if(x < m && y >= m) {//B x D^-1
            inv[idx(x, y, n)] = 0;
            for(int j = m; j < n; j++) inv[idx(x, y, n)] += g[idx(x, j, n)] * inv[idx(j, y, n)];
        }
        if(x >= m && y < m) {//D^-1 x C
            inv[idx(x, y, n)] = 0;
            for(int j = m; j < n; j++) inv[idx(x, y, n)] += inv[idx(x, j, n)] * g[idx(j, y, n)];
        }
    }
}
__global__ void build_RC_invA() {//A-BD^{-1}C
    int net_id = blockIdx.x;
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    double *g = RC_g + RC_node2_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < m * m; i++) {
        int x = (i * blockDim.x + threadIdx.x) / m, y = (i * blockDim.x + threadIdx.x) % m;
        inv[idx(x, y, n)] = g[idx(x, y, n)];
        for(int j = m; j < n; j++) inv[idx(x, y, n)] -= inv[idx(x, j, n)] * g[idx(j, y, n)];
    }
}

template<class T>
T* alloc_cpy(T *vec_cpu, int n) {
    T* vec_gpu;
    cudaMalloc(&vec_gpu, n * sizeof(T));
    cudaMemcpy(vec_gpu, vec_cpu, n * sizeof(T), cudaMemcpyHostToDevice);
    return vec_gpu;
}

__global__ void prebuild_RC_invD(int *offset, int *net, int tot_inode2, int pivot, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot_inode2) return;
    int net_id = net[idx];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    pivot += m;
    if(pivot >= n) return;
    int x = (idx - offset[idx]) / (n - m) + m, y = (idx - offset[idx]) % (n - m) + m;
    assert(x < n);
    double *g = RC_g + RC_node2_acc_num[net_id], *inv = RC_inv + RC_node2_acc_num[net_id];
    if(type == 0) {
        if(x > pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        }
    } else if(type == 1) {
        if(x < pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        } 
    } else if(type == 2) {
        inv[idx(x, y, n)] /= g[idx(x, x, n)];
    }    
}

void design:: prebuild() {

    double t = clock();
    vector<vector<int>> inode2net(max_RC_inode_count + 1);
    vector<int> inode2_sum(max_RC_inode_count + 1);
    int tot_inode2 = 0, *index2net_cpu, *index2offset_cpu;
    for(int i = 0; i < net_num; i++) if(net_type_cpu[i] > 0) {
        int inode = nets[i].n - nets[i].gates.size();
        tot_inode2 += inode * inode;
        inode2net[inode].emplace_back(i);
    }
    index2net_cpu = new int[tot_inode2]();
    index2offset_cpu = new int[tot_inode2]();
    for(int i = max_RC_inode_count, offset = 0; i >= 0; i--) {
        for(auto net_id : inode2net[i]) {
            for(int j = 0; j < i * i; j++) 
                index2net_cpu[offset + j] = net_id, index2offset_cpu[offset + j] = offset;
            offset += i * i;
        }
        inode2_sum[i] = offset;
    }
    int *index2net, *index2offset;
    cudaMalloc(&index2net, sizeof(int) * tot_inode2);
    cudaMalloc(&index2offset, sizeof(int) * tot_inode2);
    cudaMemcpy(index2net, index2net_cpu, sizeof(int) * tot_inode2, cudaMemcpyHostToDevice);
    cudaMemcpy(index2offset, index2offset_cpu, sizeof(int) * tot_inode2, cudaMemcpyHostToDevice);
    build_RC_g<<<net_num, 1024>>> ();
    for(int i = 0; i < max_RC_inode_count; i++) if(inode2_sum[i + 1] > 0)
        prebuild_RC_invD<<<BLOCK_NUM(inode2_sum[i + 1]), THREAD_NUM>>> (index2offset, index2net, inode2_sum[i + 1], i, 0);
    for(int i = max_RC_inode_count - 1; i >= 0; i--) if(inode2_sum[i + 1] > 0)
        prebuild_RC_invD<<<BLOCK_NUM(inode2_sum[i + 1]), THREAD_NUM>>> (index2offset, index2net, inode2_sum[i + 1], i, 1);
    prebuild_RC_invD<<<BLOCK_NUM(inode2_sum[0]), THREAD_NUM>>> (index2offset, index2net, inode2_sum[0], 0, 2);
    build_RC_invBC<<<net_num, 1024>>> ();
    build_RC_invA<<<net_num, 1024>>> ();
    cudaDeviceSynchronize();

    delete index2net_cpu;
    delete index2offset_cpu;
    //cudaFree(index2net);
    //cudaFree(index2offset);
}

void design::init_database_GPU() {
    max_RC_node_count = max_RC_port_count = max_RC_inode_count = input_num = output_num = 0;
    for(auto &net : nets) {
        max_RC_node_count = max(max_RC_node_count, net.n);
        max_RC_port_count = max(max_RC_port_count, int(net.gates.size()));
        max_RC_inode_count = max(max_RC_inode_count, net.n - int(net.gates.size()));
        input_num += (net.net_type == INPUT);
        output_num += (net.net_type == OUTPUT);
    }

    net_num = nets.size();
    cell_num = CCS::cells.size();
    gate_num = gates.size();


    cudaMalloc(&output_delay, sizeof(double) * net_num * 2);

    net_type_cpu = new int[net_num]();
    input_nets_cpu = new int[input_num]();
    for(int i = 0, input_cnt = 0; i < net_num; i++) {
        net_type_cpu[i] = 1;
        if(nets[i].net_type == INPUT) net_type_cpu[i] = 0, input_nets_cpu[input_cnt++] = i;
        if(nets[i].net_type == OUTPUT) net_type_cpu[i] = 2;
    }
    sort(input_nets_cpu, input_nets_cpu + input_num, [&] (int l, int r) {
        return nets[l].n > nets[r].n;
    });
    cudaMalloc(&net_type, net_num * sizeof(int));
    cudaMalloc(&input_nets, input_num * sizeof(int));
    cudaMemcpy(net_type, net_type_cpu, net_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(input_nets, input_nets_cpu, input_num * sizeof(int), cudaMemcpyHostToDevice);

    RC_port_acc_num_cpu = new int[net_num + 1]();
    RC_node_acc_num_cpu = new int[net_num + 1]();
    RC_node2_acc_num_cpu = new int[net_num + 1]();
    RC_inode2_acc_num_cpu = new int[net_num + 1]();
    RC_portinode_acc_num_cpu = new int[net_num + 1]();
    RC_input_node2_acc_num_cpu = new int[net_num + 1]();
    for(int i = 1; i <= net_num; i++) {
        int n = nets[i - 1].n, m = nets[i - 1].gates.size();
        RC_port_acc_num_cpu[i] = RC_port_acc_num_cpu[i - 1] + m;
        RC_node_acc_num_cpu[i] = RC_node_acc_num_cpu[i - 1] + n;
        RC_node2_acc_num_cpu[i] = RC_node2_acc_num_cpu[i - 1] + n * n;
        RC_inode2_acc_num_cpu[i] = RC_inode2_acc_num_cpu[i - 1] + (n - m) * (n - m);
        RC_portinode_acc_num_cpu[i] = RC_portinode_acc_num_cpu[i - 1] + m * (n - m);
        n += (nets[i - 1].net_type == INPUT);
        RC_input_node2_acc_num_cpu[i] = RC_input_node2_acc_num_cpu[i - 1] + n * n;
    }
    cudaMalloc(&RC_port_acc_num, (net_num + 1) * sizeof(int));
    cudaMalloc(&RC_node_acc_num, (net_num + 1) * sizeof(int));
    cudaMalloc(&RC_node2_acc_num, (net_num + 1) * sizeof(int));
    cudaMalloc(&RC_inode2_acc_num, (net_num + 1) * sizeof(int));
    cudaMalloc(&RC_portinode_acc_num, (net_num + 1) * sizeof(int));
    cudaMalloc(&RC_input_node2_acc_num, (net_num + 1) * sizeof(int));
    //cudaMalloc(&sorted_RC_inode2_acc_num, (net_num + 1) * sizeof(int));
    cudaMemcpy(RC_port_acc_num, RC_port_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_node_acc_num, RC_node_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_node2_acc_num, RC_node2_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_inode2_acc_num, RC_inode2_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_portinode_acc_num, RC_portinode_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_input_node2_acc_num, RC_input_node2_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(sorted_RC_inode2_acc_num, sorted_RC_inode2_acc_num_cpu, (net_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&RC_port_delay, RC_port_acc_num_cpu[net_num] * 2 * sizeof(double));

    inode2_acc_num_to_net_cpu = new int[RC_inode2_acc_num_cpu[net_num]]();
    for(int i = 0; i < net_num; i++) {
        int n = nets[i].n - nets[i].gates.size();
        for(int j = 0; j < n * n; j++) inode2_acc_num_to_net_cpu[j + RC_inode2_acc_num_cpu[i]] = i;

    }
    cudaMalloc(&inode2_acc_num_to_net, RC_inode2_acc_num_cpu[net_num] * sizeof(int));
    cudaMemcpy(inode2_acc_num_to_net, inode2_acc_num_to_net_cpu, RC_inode2_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&basic_mat, RC_node2_acc_num_cpu[net_num] * sizeof(double));
    cudaMalloc(&RC_g, RC_node2_acc_num_cpu[net_num] * sizeof(double));
    cudaMalloc(&RC_inv, RC_node2_acc_num_cpu[net_num] * sizeof(double));
    cudaMalloc(&RC_cap_g, RC_node_acc_num_cpu[net_num] * sizeof(double));

    RC_port_gate_id_cpu = new int[RC_port_acc_num_cpu[net_num]]();
    RC_port_cell_id_cpu = new int[RC_port_acc_num_cpu[net_num]]();
    RC_port_pin_id_cpu = new int[RC_port_acc_num_cpu[net_num]]();
    for(int i = 0; i < net_num; i++)
        for(int j = 0; j < nets[i].gates.size(); j++) {
            RC_port_gate_id_cpu[RC_port_acc_num_cpu[i] + j] = nets[i].gates[j].first;
            RC_port_cell_id_cpu[RC_port_acc_num_cpu[i] + j] = gates[nets[i].gates[j].first].cell_id;// starting from 1 in GPU
            RC_port_pin_id_cpu[RC_port_acc_num_cpu[i] + j] = nets[i].gates[j].second;
        }
    cudaMalloc(&RC_port_gate_id, RC_port_acc_num_cpu[net_num] * sizeof(int));
    cudaMalloc(&RC_port_cell_id, RC_port_acc_num_cpu[net_num] * sizeof(int));
    cudaMalloc(&RC_port_pin_id, RC_port_acc_num_cpu[net_num] * sizeof(int));
    cudaMemcpy(RC_port_gate_id, RC_port_gate_id_cpu, RC_port_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_port_cell_id, RC_port_cell_id_cpu, RC_port_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_port_pin_id, RC_port_pin_id_cpu, RC_port_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    
    RC_node_cap_cpu = new double[RC_node_acc_num_cpu[net_num]]();
    RC_res_node0_cpu = new int[RC_node_acc_num_cpu[net_num]]();
    RC_res_node1_cpu = new int[RC_node_acc_num_cpu[net_num]]();
    double *RC_res_cpu = new double[RC_node_acc_num_cpu[net_num]]();
    for(int i = 0; i < net_num; i++)
        for(int j = 0; j < nets[i].n; j++) {
            RC_node_cap_cpu[RC_node_acc_num_cpu[i] + j] = nets[i].caps[j];
            if(j < nets[i].ress.size()) {
                RC_res_node0_cpu[RC_node_acc_num_cpu[i] + j] = get<0> (nets[i].ress[j]);
                RC_res_node1_cpu[RC_node_acc_num_cpu[i] + j] = get<1> (nets[i].ress[j]);
                RC_res_cpu[RC_node_acc_num_cpu[i] + j] = get<2> (nets[i].ress[j]);
            } else
                assert(RC_res_cpu[RC_node_acc_num_cpu[i] + j] == 0);
        }
    cudaMalloc(&RC_node_cap, RC_node_acc_num_cpu[net_num] * sizeof(double));
    cudaMalloc(&RC_res_node0, RC_node_acc_num_cpu[net_num] * sizeof(int));
    cudaMalloc(&RC_res_node1, RC_node_acc_num_cpu[net_num] * sizeof(int));
    cudaMalloc(&RC_res, RC_node_acc_num_cpu[net_num] * sizeof(double));
    cudaMemcpy(RC_node_cap, RC_node_cap_cpu, RC_node_acc_num_cpu[net_num] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_res_node0, RC_res_node0_cpu, RC_node_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_res_node1, RC_res_node1_cpu, RC_node_acc_num_cpu[net_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(RC_res, RC_res_cpu, RC_node_acc_num_cpu[net_num] * sizeof(double), cudaMemcpyHostToDevice);


    cell_ipin_acc_num_cpu = new int[cell_num + 1]();
    for(int i = 1; i <= cell_num; i++) cell_ipin_acc_num_cpu[i] = cell_ipin_acc_num_cpu[i - 1] + CCS::cells[i - 1].i_pin_name.size();
    cudaMalloc(&cell_ipin_acc_num, (cell_num + 1) * sizeof(int));
    cudaMemcpy(cell_ipin_acc_num, cell_ipin_acc_num_cpu, (cell_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    gate_ipin_acc_num_cpu = new int[gate_num + 1]();
    for(int i = 1; i <= gate_num; i++) gate_ipin_acc_num_cpu[i] = gate_ipin_acc_num_cpu[i - 1] + CCS::cells[gates[i - 1].cell_id].i_pin_name.size();
    cudaMalloc(&gate_ipin_acc_num, (gate_num + 1) * sizeof(int));
    cudaMemcpy(gate_ipin_acc_num, gate_ipin_acc_num_cpu, (gate_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    inslew_cpu = new double[2 * gate_ipin_acc_num_cpu[gate_num]]();

    cudaMalloc(&inslew, 2 * gate_ipin_acc_num_cpu[gate_num] * sizeof(double));
    cudaMalloc(&gate_ipin_delay, 2 * gate_ipin_acc_num_cpu[gate_num] * sizeof(double));

    cell_ipin_nldm_cap_cpu = new double[2 * cell_ipin_acc_num_cpu[cell_num]]();
    for(int s = 0; s < 2; s++)
        for(int i = 0; i < cell_num; i++)
            for(int j = 0; j < CCS::cells[i].i_pin_name.size(); j++)
                cell_ipin_nldm_cap_cpu[s * cell_ipin_acc_num_cpu[cell_num] + cell_ipin_acc_num_cpu[i] + j] = CCS::cells[i].nldm_cap[s][j];
    cudaMalloc(&cell_ipin_nldm_cap, 2 * cell_ipin_acc_num_cpu[cell_num] * sizeof(double));
    cudaMemcpy(cell_ipin_nldm_cap, cell_ipin_nldm_cap_cpu, 2 * cell_ipin_acc_num_cpu[cell_num] * sizeof(double), cudaMemcpyHostToDevice);


    arc_acc_num_cpu = new int[cell_num + 1]();
    for(int i = 1; i <= cell_num; i++) {
        arc_acc_num_cpu[i] = arc_acc_num_cpu[i - 1];
        for(auto &arcs : CCS::cells[i - 1].arcs) arc_acc_num_cpu[i] += arcs.size() * 2;
    }
    ccs_acc_ilen_cpu = new int[arc_acc_num_cpu[cell_num] + 1]();
    ccs_acc_olen_cpu = new int[arc_acc_num_cpu[cell_num] + 1]();
    ccs_acc_tlen_cpu = new int[arc_acc_num_cpu[cell_num] + 1]();
    int cur = 0;
    for(auto &cell : CCS::cells) for(auto &arcs : cell.arcs) for(auto &arc : arcs) for(int s = 0; s < 2; s++) {
        cur++;
        auto &table = arc.current[s];
        ccs_acc_ilen_cpu[cur] = ccs_acc_ilen_cpu[cur - 1] + table.input_slew.size();
        ccs_acc_olen_cpu[cur] = ccs_acc_olen_cpu[cur - 1] + table.output_cap.size() - 2;// output_cap was expanded
        ccs_acc_tlen_cpu[cur] = ccs_acc_tlen_cpu[cur - 1] + table.input_slew.size() * (table.output_cap.size() - 2) * VPLEN * 3;
    }
    cudaMalloc(&ccs_acc_ilen, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int));
    cudaMalloc(&ccs_acc_olen, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int));
    cudaMalloc(&ccs_acc_tlen, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int));
    cudaMemcpy(ccs_acc_ilen, ccs_acc_ilen_cpu, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ccs_acc_olen, ccs_acc_olen_cpu, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ccs_acc_tlen, ccs_acc_tlen_cpu, (arc_acc_num_cpu[cell_num] + 1) * sizeof(int), cudaMemcpyHostToDevice);


    ccs_refer_time_cpu = new double[ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]]]();
    ccs_input_slew_cpu = new double[ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]]]();
    ccs_output_cap_cpu = new double[ccs_acc_olen_cpu[arc_acc_num_cpu[cell_num]]]();
    ccs_tp_cpu = new double[ccs_acc_tlen_cpu[arc_acc_num_cpu[cell_num]]]();
    cur = 0;
    for(auto &cell : CCS::cells) for(auto &arcs : cell.arcs) for(auto &arc : arcs) for(int s = 0; s < 2; s++) {
        auto &table = arc.current[s];
        assert(ccs_acc_ilen_cpu[cur + 1] - ccs_acc_ilen_cpu[cur] == table.input_slew.size());
        assert(ccs_acc_olen_cpu[cur + 1] - ccs_acc_olen_cpu[cur] == table.output_cap.size() - 2);
        for(int i = 0; i < table.input_slew.size(); i++) {
            ccs_refer_time_cpu[ccs_acc_ilen_cpu[cur] + i] = table.refer_time[i];
            ccs_input_slew_cpu[ccs_acc_ilen_cpu[cur] + i] = table.input_slew[i];
        }
        for(int i = 0; i < table.output_cap.size() - 2; i++) 
            ccs_output_cap_cpu[ccs_acc_olen_cpu[cur] + i] = table.output_cap[i + 1];
        for(int t = 0, cnt = 0; t < 3; t++) {
            for(int i = 0; i < table.input_slew.size(); i++)
                for(int j = 0; j < table.output_cap.size() - 2; j++)
                    for(int k = 0; k < VPLEN; k++)
                        ccs_tp_cpu[ccs_acc_tlen_cpu[cur] + cnt++] = table.V_times[t][i][j][k];
        }
        table.id = cur++;
    }
    cudaMalloc(&ccs_refer_time, ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double));
    cudaMalloc(&ccs_input_slew, ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double));
    cudaMalloc(&ccs_output_cap, ccs_acc_olen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double));
    cudaMalloc(&ccs_tp, ccs_acc_tlen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double));
    cudaMemcpy(ccs_refer_time, ccs_refer_time_cpu, ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ccs_input_slew, ccs_input_slew_cpu, ccs_acc_ilen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ccs_output_cap, ccs_output_cap_cpu, ccs_acc_olen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ccs_tp, ccs_tp_cpu, ccs_acc_tlen_cpu[arc_acc_num_cpu[cell_num]] * sizeof(double), cudaMemcpyHostToDevice);
    
    assert(cudaGetLastError() == cudaSuccess);
    prebuild();

    /*int max_output_cap_len = 0;
    for(int i = 1; i <= arc_acc_num_cpu[cell_num]; i++)
        max_output_cap_len = max(max_output_cap_len, ccs_acc_olen_cpu[i] - ccs_acc_olen_cpu[i - 1]);
    output_log("max_output_cap_len = " + to_string(max_output_cap_len));*/
}

__global__ void node2_to_stage() {
    int stage_id = blockIdx.x, tot = stage_node2_acc_num[stage_id + 1] - stage_node2_acc_num[stage_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < tot; i++)
        node2_acc_num_to_stage[stage_node2_acc_num[stage_id] + i * blockDim.x + threadIdx.x] = stage_id;
}
__global__ void port2_to_stage() {
    int stage_id = blockIdx.x, tot = stage_port2_acc_num[stage_id + 1] - stage_port2_acc_num[stage_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < tot; i++)
        port2_acc_num_to_stage[stage_port2_acc_num[stage_id] + i * blockDim.x + threadIdx.x] = stage_id;
}
__global__ void portinode_to_stage() {
    int stage_id = blockIdx.x, tot = stage_portinode_acc_num[stage_id + 1] - stage_portinode_acc_num[stage_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < tot; i++)
        portinode_acc_num_to_stage[stage_portinode_acc_num[stage_id] + i * blockDim.x + threadIdx.x] = stage_id;
}
__global__ void inode2_to_stage() {
    int stage_id = blockIdx.x, tot = stage_inode2_acc_num[stage_id + 1] - stage_inode2_acc_num[stage_id];
    for(int i = 0; i * blockDim.x + threadIdx.x < tot; i++)
        inode2_acc_num_to_stage[stage_inode2_acc_num[stage_id] + i * blockDim.x + threadIdx.x] = stage_id;
}


const vector<int> sizes = {0, 32, 1024};
void design::init_graph_GPU() {
    double t = clock();
    topo = levelize();
    level_stage_acc_num = vector<int> (topo.size() + 1, 0);
    for(int i = 1; i <= topo.size(); i++) {
        level_stage_acc_num[i] = level_stage_acc_num[i - 1];
        for(auto gate_id : topo[i - 1]) {
            auto &cell = CCS::cells[gates[gate_id].cell_id];
            for(int j = 0; j < cell.get_arc_size(); j++)
                for(auto &arc : cell.arcs[j]) for(int s = 0; s < 4; s++) if(arc.signal[s]) level_stage_acc_num[i]++;
        }
    }
    stage_num = level_stage_acc_num[topo.size()];

    
    cerr << setw(10) << "#PIs" << setw(10) << input_num << endl;
    cerr << setw(10) << "#POs" << setw(10) << input_num << endl;
    cerr << setw(10) << "#Gates" << setw(10) << gate_num << endl;
    cerr << setw(10) << "#Nets" << setw(10) << net_num << endl;
    cerr << setw(10) << "#Stages" << setw(10) << stage_num << endl;
    double mn = 1000, mx = 0, avg = 0;
    for(int i = 0; i < net_num; i++) {
        mn = min(mn, 1.0 * nets[i].n);
        mx = max(mx, 1.0 * nets[i].n);
        avg += nets[i].n;
    }
    cerr << setw(10) << "min" << setw(10) << mn << endl;
    cerr << setw(10) << "avg" << setw(10) << avg / net_num << endl;
    cerr << setw(10) << "max" << setw(10) << mx << endl;
    //cerr << setw(10) << "#stages" << setw(10) << stage_num << endl;
    //cerr << setw(10) << "max #nodes" << setw(10) << max_RC_node_count << endl;

    size_num = size_max_node_num = size_offset = vector<vector<int>> (topo.size(), vector<int> (sizes.size()));

    stage_driver_gate_id_cpu = new int[stage_num + 1]();
    stage_driver_ipin_cpu = new int[stage_num + 1]();
    stage_signal_cpu = new int[stage_num + 1]();
    stage_net_id_cpu = new int[stage_num + 1]();
    stage_arc_id_cpu = new int[stage_num + 1]();
    stage_node_acc_num_cpu = new int[stage_num + 1]();
    stage_node2_acc_num_cpu = new int[stage_num + 1]();
    stage_port2_acc_num_cpu = new int[stage_num + 1]();
    stage_portinode_acc_num_cpu = new int[stage_num + 1]();
    stage_inode2_acc_num_cpu = new int[stage_num + 1]();
    stage_port_acc_num_cpu = new int[stage_num + 1]();
    for(int i = 0, cur = 0; i < topo.size(); i++) {
        for(int iter = 1; iter < sizes.size(); iter++) {
            size_offset[i][iter] = cur;
            size_max_node_num[i][iter] = 0;
            for(auto &gate_id : topo[i]) {
                auto &cell = CCS::cells[gates[gate_id].cell_id];
                for(int arc_id = 0; arc_id < cell.get_arc_size(); arc_id++) {
                    int ipin = arc_id / cell.get_opin_size(), opin = arc_id % cell.get_opin_size();
                    for(auto &arc : cell.arcs[arc_id]) for(int s = 0; s < 4; s++) if(arc.signal[s]) {
                        int n = nets[gates[gate_id].o_nets[opin]].n, m = nets[gates[gate_id].o_nets[opin]].gates.size();
                        if(!(sizes[iter - 1] < n && n <= sizes[iter])) continue;
                        stage_driver_gate_id_cpu[cur] = gate_id;
                        stage_driver_ipin_cpu[cur] = ipin;
                        stage_signal_cpu[cur] = s;
                        stage_net_id_cpu[cur] = gates[gate_id].o_nets[opin];
                        stage_arc_id_cpu[cur] = arc.current[s % 2].id;
                        stage_node_acc_num_cpu[cur + 1] = stage_node_acc_num_cpu[cur] + n;
                        stage_node2_acc_num_cpu[cur + 1] = stage_node2_acc_num_cpu[cur] + n * n;
                        stage_port2_acc_num_cpu[cur + 1] = stage_port2_acc_num_cpu[cur] + m * m;
                        stage_portinode_acc_num_cpu[cur + 1] = stage_portinode_acc_num_cpu[cur] + m * (n - m);
                        stage_inode2_acc_num_cpu[cur + 1] = stage_inode2_acc_num_cpu[cur] + (n - m) * (n - m);
                        stage_port_acc_num_cpu[cur + 1] = stage_port_acc_num_cpu[cur] + m;
                        size_max_node_num[i][iter] = max(size_max_node_num[i][iter], n);
                        cur++;
                    }
                }
            }
            size_num[i][iter] = cur - size_offset[i][iter];
        }
    }
    cudaMalloc(&stage_driver_gate_id, stage_num * sizeof(int));
    cudaMalloc(&stage_driver_ipin, stage_num * sizeof(int));
    cudaMalloc(&stage_signal, stage_num * sizeof(int));
    cudaMalloc(&stage_net_id, stage_num * sizeof(int));
    cudaMalloc(&stage_arc_id, stage_num * sizeof(int));
    cudaMalloc(&stage_node2_acc_num, (stage_num + 1) * sizeof(int));
    cudaMalloc(&stage_port2_acc_num, (stage_num + 1) * sizeof(int));
    cudaMalloc(&stage_node_acc_num, (stage_num + 1) * sizeof(int));
    cudaMalloc(&stage_portinode_acc_num, (stage_num + 1) * sizeof(int));
    cudaMalloc(&stage_inode2_acc_num, (stage_num + 1) * sizeof(int));
    cudaMalloc(&stage_port_acc_num, (stage_num + 1) * sizeof(int));
    cudaMemcpy(stage_driver_gate_id, stage_driver_gate_id_cpu, stage_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_driver_ipin, stage_driver_ipin_cpu, stage_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_signal, stage_signal_cpu, stage_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_net_id, stage_net_id_cpu, stage_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_arc_id, stage_arc_id_cpu, stage_num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_node_acc_num, stage_node_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_node2_acc_num, stage_node2_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_port2_acc_num, stage_port2_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_portinode_acc_num, stage_portinode_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_inode2_acc_num, stage_inode2_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(stage_port_acc_num, stage_port_acc_num_cpu, (stage_num + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&stage_delay, stage_port_acc_num_cpu[stage_num] * sizeof(double));
    cudaMalloc(&stage_port_cap, 3 * stage_port_acc_num_cpu[stage_num] * sizeof(double));

    cudaMalloc(&node2_acc_num_to_stage, stage_node2_acc_num_cpu[stage_num] * sizeof(int));
    cudaMalloc(&port2_acc_num_to_stage, stage_port2_acc_num_cpu[stage_num] * sizeof(int));
    cudaMalloc(&portinode_acc_num_to_stage, stage_portinode_acc_num_cpu[stage_num] * sizeof(int));
    cudaMalloc(&inode2_acc_num_to_stage, stage_inode2_acc_num_cpu[stage_num] * sizeof(int));

    int temp = stage_inode2_acc_num_cpu[stage_num] + stage_portinode_acc_num_cpu[stage_num] + 
        stage_port2_acc_num_cpu[stage_num] + stage_node2_acc_num_cpu[stage_num];
    cerr << temp * sizeof(int) / 1024 / 1024 / 1024 << "GB" << endl;

    /*node2_acc_num_to_stage_cpu = new int[stage_node2_acc_num_cpu[stage_num]]();
    port2_acc_num_to_stage_cpu = new int[stage_port2_acc_num_cpu[stage_num]]();
    portinode_acc_num_to_stage_cpu = new int[stage_portinode_acc_num_cpu[stage_num]]();
    inode2_acc_num_to_stage_cpu = new int[stage_inode2_acc_num_cpu[stage_num]]();
    printf("temp time0 = %.4f\n", (clock() - t) / CLOCKS_PER_SEC);
    for(int i = 0; i < stage_num; i++) {
        //for(int j = stage_node2_acc_num_cpu[i]; j < stage_node2_acc_num_cpu[i + 1]; j++) node2_acc_num_to_stage_cpu[j] = i;
        for(int j = stage_port2_acc_num_cpu[i]; j < stage_port2_acc_num_cpu[i + 1]; j++) port2_acc_num_to_stage_cpu[j] = i;
        for(int j = stage_portinode_acc_num_cpu[i]; j < stage_portinode_acc_num_cpu[i + 1]; j++) portinode_acc_num_to_stage_cpu[j] = i;
        for(int j = stage_inode2_acc_num_cpu[i]; j < stage_inode2_acc_num_cpu[i + 1]; j++) inode2_acc_num_to_stage_cpu[j] = i;
    }
    printf("temp time0.5 = %.4f\n", (clock() - t) / CLOCKS_PER_SEC);
    cudaMemcpy(node2_acc_num_to_stage, node2_acc_num_to_stage_cpu, stage_node2_acc_num_cpu[stage_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(port2_acc_num_to_stage, port2_acc_num_to_stage_cpu, stage_port2_acc_num_cpu[stage_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(portinode_acc_num_to_stage, portinode_acc_num_to_stage_cpu, stage_portinode_acc_num_cpu[stage_num] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(inode2_acc_num_to_stage, inode2_acc_num_to_stage_cpu, stage_inode2_acc_num_cpu[stage_num] * sizeof(int), cudaMemcpyHostToDevice);
    */
    node2_to_stage<<<stage_num, 1024>>> ();
    port2_to_stage<<<stage_num, 1024>>> ();
    portinode_to_stage<<<stage_num, 1024>>> ();
    inode2_to_stage<<<stage_num, 1024>>> ();


    cudaMalloc(&stage_cap_g, (INV3 ? 3 : 2) * stage_node_acc_num_cpu[stage_num] * sizeof(double));
    cudaMalloc(&stage_g, (INV3 ? 3 : 2) * stage_node2_acc_num_cpu[stage_num] * sizeof(double));
    cudaMalloc(&stage_inv, (INV3 ? 3 : 2) * stage_node2_acc_num_cpu[stage_num] * sizeof(double));

    cerr << stage_node2_acc_num_cpu[stage_num] / 1024.0 / 1024 / 1024 * 6 * sizeof(double) << "GB" << endl;

    RC_ccs_cap_cpu = new double[RC_port_acc_num_cpu[net_num] * TABLE_LEN * 4]();
    for(int i = 0; i < net_num; i++) {
        auto &net = nets[i];
        for(int j = 1; j < net.gates.size(); j++) {
            const auto &gate = gates[net.gates[j].first];
            auto &cell = CCS::cells[gate.cell_id];
            for(int signal = 0; signal < 2; signal++)
                for(int iter = 0; iter < 2; iter++) {
                    vector<double> max_cap(TABLE_LEN, 0);
                    for(int opin = 0; opin < cell.get_opin_size(); opin++) {
                        int arc_idx = cell.get_arc_index(net.gates[j].second, opin);
                        for(auto &arc : cell.arcs[arc_idx]) {
                            auto temp = arc.ccs_cap[signal][iter].lookup_cap(nets[gate.o_nets[opin]].total_cap);
                            assert(temp.size() == TABLE_LEN);
                            double tot = 0;
                            for(int k = 0; k < TABLE_LEN; k++) tot += temp[k] - max_cap[k];
                            if(tot > 0) max_cap = temp;
                        }
                    }
                    for(int k = 0; k < TABLE_LEN; k++)
                        RC_ccs_cap_cpu[((RC_port_acc_num_cpu[i] + j) * 4 + signal * 2 + iter) * TABLE_LEN + k] = max_cap[k];
                }
        }
    }
    cudaMalloc(&RC_ccs_cap, RC_port_acc_num_cpu[net_num] * TABLE_LEN * 4 * sizeof(double));
    cudaMemcpy(RC_ccs_cap, RC_ccs_cap_cpu, RC_port_acc_num_cpu[net_num] * TABLE_LEN * 4 * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    printf("[END] init graph. error = %d\n", cudaGetLastError());
}


__global__ void build_basic_mat() {
    int net_id = blockIdx.x, node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    double *g = basic_mat + RC_node2_acc_num[net_id];

    for(int i = 0; i * blockDim.x + threadIdx.x < n * n; i++) g[i * blockDim.x + threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < n && RC_res[node_offset + threadIdx.x] > 0) {
        int u = RC_res_node0[node_offset + threadIdx.x], v = RC_res_node1[node_offset + threadIdx.x];
        double conductance = 1.0 / RC_res[node_offset + threadIdx.x];
        atomicAdd(g + idx(u, u, n), conductance);
        atomicAdd(g + idx(v, v, n), conductance);
        atomicAdd(g + idx(u, v, n), -conductance);
        atomicAdd(g + idx(v, u, n), -conductance);
    }
}
__global__ void build_input_mat(int *input_node2_acc_num, int input_num) {
    int signal = blockIdx.x / input_num;
    assert(signal < 2);
    int net_id = input_nets[blockIdx.x % input_num];
    int RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    int node2_offset = signal * input_node2_acc_num[input_num] + input_node2_acc_num[blockIdx.x % input_num];
    const double deltaT = DELTA_T;
    double *g = input_g + node2_offset, *inv = input_inv + node2_offset;
    double cap_g = 0;

    for(int i = 0; i * blockDim.x + threadIdx.x < (n + 1) * (n + 1); i++) 
        g[i * blockDim.x + threadIdx.x] = inv[i * blockDim.x + threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < n) cap_g = RC_node_cap[RC_node_offset + threadIdx.x];
    if(threadIdx.x <= n) inv[idx(threadIdx.x, threadIdx.x, n + 1)] = 1;
    __syncthreads();
    if(1 <= threadIdx.x && threadIdx.x < m) cap_g += cell_ipin_nldm_cap[cell_ipin_acc_num[cell_num] * signal +
        cell_ipin_acc_num[RC_port_cell_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x]];
    if(threadIdx.x < n) {
        g[idx(threadIdx.x, threadIdx.x, n + 1)] += (cap_g *= 2 / deltaT);
        input_cap_g[node2_offset + threadIdx.x] = cap_g;
    }
    __syncthreads();
    if(threadIdx.x < n && RC_res[RC_node_offset + threadIdx.x] > 0) {
        int u = RC_res_node0[RC_node_offset + threadIdx.x], v = RC_res_node1[RC_node_offset + threadIdx.x];
        double conductance = 1.0 / RC_res[RC_node_offset + threadIdx.x];
        atomicAdd(g + idx(u, u, n + 1), conductance);
        atomicAdd(g + idx(v, v, n + 1), conductance);
        atomicAdd(g + idx(u, v, n + 1), -conductance);
        atomicAdd(g + idx(v, u, n + 1), -conductance);
    }
    if(threadIdx.x == 0) g[idx(n, 0, n + 1)] = g[idx(0, n, n + 1)] = 1;
}

__global__ void build_input_inv(int *input_node2_acc_num, int *idx2net, int tot_node2, int tot, int pivot, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= 2 * tot_node2) return;
    int signal = idx / tot_node2;
    assert(signal < 2);
    int net_index = idx2net[idx % tot_node2], net_id = input_nets[net_index], n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id] + 1;
    if(pivot >= n) return;
    idx -= input_node2_acc_num[net_index] + signal * tot_node2;
    int x = idx / n, y = idx % n;
    assert(x < n);
    double *g = input_g + signal * tot + input_node2_acc_num[net_index];
    double *inv = input_inv + signal * tot + input_node2_acc_num[net_index];
    if(type == 0) {
        if(x > pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        }
    } else if(type == 1) {
        if(x < pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        } 
    } else if(type == 2) {
        inv[idx(x, y, n)] /= g[idx(x, x, n)];
    }    
}
//const bool INV3 = false;
__global__ void build_stage_matA(int pass, int offset = 0) {//compute A - BD^{-1}C
    int stage_id = blockIdx.x + offset, net_id = stage_net_id[stage_id];
    int RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    double *g1 = stage_g + stage_node2_acc_num[stage_id], *g2 = g1 + stage_node2_acc_num[stage_num], *g3 = g2 + stage_node2_acc_num[stage_num];
    double *inv1 = stage_inv + stage_node2_acc_num[stage_id], *inv2 = inv1 + stage_node2_acc_num[stage_num], *inv3 = inv2 + stage_node2_acc_num[stage_num];
    double *cap_g1 = stage_cap_g + stage_node_acc_num[stage_id], *cap_g2 = cap_g1 + stage_node_acc_num[stage_num], *cap_g3 = cap_g2 + stage_node_acc_num[stage_num];
    for(int i = 0; i * blockDim.x + threadIdx.x < m * m; i++) {
        int idx = i * blockDim.x + threadIdx.x, x = idx / m, y = idx % m;
        g1[idx(x, y, n)] = g2[idx(x, y, n)] = RC_inv[RC_node2_acc_num[net_id] + idx(x, y, n)];//A-BD^{-1}C (A is partial, and will be complete with the following lines)
        inv1[idx(x, y, n)] = inv2[idx(x, y, n)] = (x == y ? 1 : 0);
        if(INV3) {
            g3[idx(x, y, n)] = RC_inv[RC_node2_acc_num[net_id] + idx(x, y, n)];
            inv3[idx(x, y, n)] = (x == y ? 1 : 0);
        }
    }
    __syncthreads();
    if(threadIdx.x < n) {
        cap_g1[threadIdx.x] = cap_g2[threadIdx.x] = RC_cap_g[RC_node_offset + threadIdx.x];
        if(INV3) cap_g3[threadIdx.x] = RC_cap_g[RC_node_offset + threadIdx.x];
    }
    if(1 <= threadIdx.x && threadIdx.x < m) {
        double cap1 = 0, cap2 = 0, cap3 = 0;
        if(pass == 0) cap1 = cap2 = cap3 = cell_ipin_nldm_cap[cell_ipin_acc_num[cell_num] * (stage_signal[stage_id] % 2) +
            cell_ipin_acc_num[RC_port_cell_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x]];
        if(pass == 1) {
            cap1 = stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3];
            cap2 = stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 1];
            if(INV3) cap3 = stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 2];
        }
        cap1 *= 2 / DELTA_T;
        cap2 *= 2 / DELTA_T;
        if(INV3) cap3 *= 2 / DELTA_T;
        g1[idx(threadIdx.x, threadIdx.x, n)] += cap1;
        g2[idx(threadIdx.x, threadIdx.x, n)] += cap2;
        if(INV3) g3[idx(threadIdx.x, threadIdx.x, n)] += cap3;
        cap_g1[threadIdx.x] += cap1;
        cap_g2[threadIdx.x] += cap2;
        if(INV3) cap_g3[threadIdx.x] += cap3;
    }
}
__global__ void build_mat_inv(int stage_node2_num_offset, int stage_node2_num, int pivot, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= stage_node2_num) return;
    int stage_id = node2_acc_num_to_stage[stage_node2_num_offset + idx], net_id = stage_net_id[stage_id];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    if(pivot >= n) return;
    idx += stage_node2_num_offset - stage_node2_acc_num[stage_id];
    int x = idx / n, y = idx % n;
    double *g = stage_g + stage_node2_acc_num[stage_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
    if(type == 0) {
        if(x > pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        }
    } else if(type == 1) {
        if(x < pivot) {
            double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
            if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
            inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
        } 
    } else if(type == 2) {
        inv[idx(x, y, n)] /= g[idx(x, x, n)];
    }    
}
__global__ void build_stage_invA(int stage_port2_num, int pivot, int type, int pass, int offset = 0) {
    //if(blockIdx.x == 0 && threadIdx.x == 0) printf("stage_port2_num=%d\n", stage_port2_num);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= stage_port2_num) return;
    idx += offset;
    int stage_id = port2_acc_num_to_stage[idx], net_id = stage_net_id[stage_id];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    if(pivot >= m) return;
    int x = (idx - stage_port2_acc_num[stage_id]) / m, y = (idx - stage_port2_acc_num[stage_id]) % m;
    if(pass == 0) {
        double *g = stage_g + stage_node2_acc_num[stage_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
        if(type == 0) {
            if(x > pivot) {
                double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
                if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
                inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
            }
        } else if(type == 1) {
            if(x < pivot) {
                double coeff = g[idx(x, pivot, n)] / g[idx(pivot, pivot, n)];
                if(y > pivot) g[idx(x, y, n)] -= coeff * g[idx(pivot, y, n)];
                inv[idx(x, y, n)] -= coeff * inv[idx(pivot, y, n)];
            } 
        } else if(type == 2) {
            inv[idx(x, y, n)] /= g[idx(x, x, n)];
        }    
    } else {
        double *g1 = stage_g + stage_node2_acc_num[stage_id], *inv1 = stage_inv + stage_node2_acc_num[stage_id];
        double *g2 = g1 + stage_node2_acc_num[stage_num], *inv2 = inv1 + stage_node2_acc_num[stage_num];
        double *g3 = g2 + stage_node2_acc_num[stage_num], *inv3 = inv2 + stage_node2_acc_num[stage_num];
        if(type == 0) {
            if(x > pivot) {
                double coeff = g1[idx(x, pivot, n)] / g1[idx(pivot, pivot, n)];
                if(y > pivot) g1[idx(x, y, n)] -= coeff * g1[idx(pivot, y, n)];
                inv1[idx(x, y, n)] -= coeff * inv1[idx(pivot, y, n)];
                coeff = g2[idx(x, pivot, n)] / g2[idx(pivot, pivot, n)];
                if(y > pivot) g2[idx(x, y, n)] -= coeff * g2[idx(pivot, y, n)];
                inv2[idx(x, y, n)] -= coeff * inv2[idx(pivot, y, n)];
                if(INV3) {
                    coeff = g3[idx(x, pivot, n)] / g3[idx(pivot, pivot, n)];
                    if(y > pivot) g3[idx(x, y, n)] -= coeff * g3[idx(pivot, y, n)];
                    inv3[idx(x, y, n)] -= coeff * inv3[idx(pivot, y, n)];
                }
            }
        } else if(type == 1) {
            if(x < pivot) {
                double coeff = g1[idx(x, pivot, n)] / g1[idx(pivot, pivot, n)];
                if(y > pivot) g1[idx(x, y, n)] -= coeff * g1[idx(pivot, y, n)];
                inv1[idx(x, y, n)] -= coeff * inv1[idx(pivot, y, n)];
                coeff = g2[idx(x, pivot, n)] / g2[idx(pivot, pivot, n)];
                if(y > pivot) g2[idx(x, y, n)] -= coeff * g2[idx(pivot, y, n)];
                inv2[idx(x, y, n)] -= coeff * inv2[idx(pivot, y, n)];
                if(INV3) {
                    coeff = g3[idx(x, pivot, n)] / g3[idx(pivot, pivot, n)];
                    if(y > pivot) g3[idx(x, y, n)] -= coeff * g3[idx(pivot, y, n)];
                    inv3[idx(x, y, n)] -= coeff * inv3[idx(pivot, y, n)];
                }
            } 
        } else if(type == 2) {
            inv1[idx(x, y, n)] /= g1[idx(x, x, n)];
            inv2[idx(x, y, n)] /= g2[idx(x, x, n)];
            if(INV3) inv3[idx(x, y, n)] /= g3[idx(x, x, n)];
        }    
    }
}
__global__ void build_stage_invBC(int tot_portinode_num, int type, int pass, int offset = 0) {//type=0: B; type=1: C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot_portinode_num) return;
    idx += offset;
    int stage_id = portinode_acc_num_to_stage[idx], net_id = stage_net_id[stage_id];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    if(pass == 0) {
        double *inv = stage_inv + stage_node2_acc_num[stage_id], *rc_inv = RC_inv + RC_node2_acc_num[net_id];
        if(type == 0) {//B: m x (n-m)
            int x = (idx - stage_portinode_acc_num[stage_id]) / (n - m), y = (idx - stage_portinode_acc_num[stage_id]) % (n - m) + m;
            inv[idx(x, y, n)] = 0;
            for(int i = 0; i < m; i++) inv[idx(x, y, n)] -= inv[idx(x, i, n)] * rc_inv[idx(i, y, n)];
        } else {//C: (n-m) x m
            int x = (idx - stage_portinode_acc_num[stage_id]) / m + m, y = (idx - stage_portinode_acc_num[stage_id]) % m;
            inv[idx(x, y, n)] = 0;
            for(int i = 0; i < m; i++) inv[idx(x, y, n)] -= rc_inv[idx(x, i, n)] * inv[idx(i, y, n)];
        } 
    } else {
        double *inv1 = stage_inv + stage_node2_acc_num[stage_id], *inv2 = inv1 + stage_node2_acc_num[stage_num], *inv3 = inv2 + stage_node2_acc_num[stage_num];
        double *rc_inv = RC_inv + RC_node2_acc_num[net_id];
        if(type == 0) {//B: m x (n-m)
            int x = (idx - stage_portinode_acc_num[stage_id]) / (n - m), y = (idx - stage_portinode_acc_num[stage_id]) % (n - m) + m;
            inv1[idx(x, y, n)] = inv2[idx(x, y, n)] = 0;
            if(INV3) inv3[idx(x, y, n)] = 0;
            for(int i = 0; i < m; i++) {
                inv1[idx(x, y, n)] -= inv1[idx(x, i, n)] * rc_inv[idx(i, y, n)];
                inv2[idx(x, y, n)] -= inv2[idx(x, i, n)] * rc_inv[idx(i, y, n)];
                if(INV3) inv3[idx(x, y, n)] -= inv3[idx(x, i, n)] * rc_inv[idx(i, y, n)];
            }
        } else {//C: (n-m) x m
            int x = (idx - stage_portinode_acc_num[stage_id]) / m + m, y = (idx - stage_portinode_acc_num[stage_id]) % m;
            inv1[idx(x, y, n)] = inv2[idx(x, y, n)] = 0;
            if(INV3) inv3[idx(x, y, n)] = 0;
            for(int i = 0; i < m; i++) {
                inv1[idx(x, y, n)] -= rc_inv[idx(x, i, n)] * inv1[idx(i, y, n)];
                inv2[idx(x, y, n)] -= rc_inv[idx(x, i, n)] * inv2[idx(i, y, n)];
                if(INV3) inv3[idx(x, y, n)] -= rc_inv[idx(x, i, n)] * inv3[idx(i, y, n)];
            }
        }         
    }
}
__global__ void build_stage_invD(int tot_inode2_num, int pass, int offset = 0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tot_inode2_num) return;
    idx += offset;
    int stage_id = inode2_acc_num_to_stage[idx], net_id = stage_net_id[stage_id];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    int m = RC_port_acc_num[net_id + 1] - RC_port_acc_num[net_id];
    if(pass == 0) {
        double *inv = stage_inv + stage_node2_acc_num[stage_id], *rc_inv = RC_inv + RC_node2_acc_num[net_id];        
        int x = (idx - stage_inode2_acc_num[stage_id]) / (n - m) + m, y = (idx - stage_inode2_acc_num[stage_id]) % (n - m) + m;
        inv[idx(x, y, n)] = rc_inv[idx(x, y, n)];
        for(int i = 0; i < m; i++) inv[idx(x, y, n)] -= inv[idx(x, i, n)] * rc_inv[idx(i, y, n)];
    } else {
        double *inv1 = stage_inv + stage_node2_acc_num[stage_id], *inv2 = inv1 + stage_node2_acc_num[stage_num], *inv3 = inv2 + stage_node2_acc_num[stage_num];
        double *rc_inv = RC_inv + RC_node2_acc_num[net_id];
        int x = (idx - stage_inode2_acc_num[stage_id]) / (n - m) + m, y = (idx - stage_inode2_acc_num[stage_id]) % (n - m) + m;
        inv1[idx(x, y, n)] = inv2[idx(x, y, n)] = rc_inv[idx(x, y, n)];
        if(INV3) inv3[idx(x, y, n)] = rc_inv[idx(x, y, n)];
        for(int i = 0; i < m; i++) {
            inv1[idx(x, y, n)] -= inv1[idx(x, i, n)] * rc_inv[idx(i, y, n)];
            inv2[idx(x, y, n)] -= inv2[idx(x, i, n)] * rc_inv[idx(i, y, n)];
            if(INV3) inv3[idx(x, y, n)] -= inv3[idx(x, i, n)] * rc_inv[idx(i, y, n)];
        }
    }
}

__global__ void print(int stage_id) {
    int net_id = stage_net_id[stage_id];
    int n = RC_node_acc_num[net_id + 1] - RC_node_acc_num[net_id];
    double *g = stage_g + stage_node2_acc_num[stage_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
    printf("print MY g \n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++)
            printf("%13.3f", g[idx(i, j, n)]);
        printf("\n");
    }    
    printf("print MY inv\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++)
            printf("%12.4f", inv[idx(i, j, n)]);
        printf("\n");
    }
}

__global__ void pass0(int stage_offset) {
    extern __shared__ double shared[];

    int stage_id = blockIdx.x + stage_offset;
    int net_id = stage_net_id[stage_id], gate_id = stage_driver_gate_id[stage_id], arc_id = stage_arc_id[stage_id];
    int ipin = stage_driver_ipin[stage_id], isignal = stage_signal[stage_id] / 2, osignal = stage_signal[stage_id] % 2;
    int RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    int olen = ccs_acc_olen[arc_id + 1] - ccs_acc_olen[arc_id];
    double *initial_tp = ccs_tp + ccs_acc_tlen[arc_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
    double *I_src = shared, *output_cap = I_src + n, *tp = output_cap + olen + 2;
    double cap_g, slew = inslew[gate_ipin_acc_num[gate_num] * isignal + gate_ipin_acc_num[gate_id] + ipin];

    if(threadIdx.x < n) cap_g = stage_cap_g[stage_node_acc_num[stage_id] + threadIdx.x];

    if(threadIdx.x == 0) output_cap[0] = ccs_output_cap[ccs_acc_olen[arc_id]] * 0.2;
    if(1 <= threadIdx.x && threadIdx.x <= olen) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + threadIdx.x - 1];
    if(threadIdx.x == olen + 1) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + olen - 1] * 1.1;

    assert(blockDim.x >= olen + 2);
    
    assert(blockDim.x >= n);
    {
        int ilen = ccs_acc_ilen[arc_id + 1] - ccs_acc_ilen[arc_id];
        __shared__ int index;
        __shared__ double coeff;
        if(threadIdx.x == 0) {
            index = 0;
            double *input_slew = ccs_input_slew + ccs_acc_ilen[arc_id];
            if(slew < input_slew[0] * 0.2) slew = input_slew[0] * 0.2;
            if(slew > input_slew[ilen - 1] * 1.1) slew = input_slew[ilen - 1] * 1.1;
            for(int i = ilen - 2; i >= 0; i--) if(input_slew[i] < slew) { index = i; break; }
            coeff = (slew - input_slew[index]) / (input_slew[index + 1] - input_slew[index]);
        }
        __syncthreads();
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN * olen; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN / olen, a = temp / olen % VPLEN, b = temp % olen;
            tp[idx3D(t, a, b + 1, VPLEN, olen + 2)] = initial_tp[idx3D(t, index, b, ilen, olen) * VPLEN + a] * (1 - coeff)
                + initial_tp[idx3D(t, index + 1, b, ilen, olen) * VPLEN + a] * coeff;
        }
        __syncthreads();
        olen += 2;
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN, a = temp % VPLEN, pos = idx3D(t, a, 0, VPLEN, olen);
            tp[pos] = tp[pos + 1] - (tp[pos + 2] - tp[pos + 1]) / (output_cap[2] - output_cap[1]) * (output_cap[1] - output_cap[0]);
            pos = idx3D(t, a, olen - 2, VPLEN, olen);
            tp[pos + 1] = tp[pos] + (tp[pos] - tp[pos - 1]) / (output_cap[olen - 2] - output_cap[olen - 3]) * (output_cap[olen - 1] - output_cap[olen - 2]);

        }
        __syncthreads();
    }


    const double deltaT = DELTA_T;

    double lastI = 0, lastV = 0.01 * VDD, curV, sim_time = tp[idx3D(1, 0, 0, VPLEN, olen)], port_slew = -1;
    __shared__ double isrc0;
    __shared__ int vt_reach_count, reach50_count;
    bool reached_vt = false, reach50 = false, early = false;
    if(threadIdx.x == 0) vt_reach_count = 0, reach50_count = 0;
    __syncthreads();
    while(1) {
        sim_time += deltaT;
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        __syncthreads();
        if(threadIdx.x < n) for(int i = curV = 0; i < n; i++) curV += inv[idx(threadIdx.x, i, n)] * I_src[i];
        if(threadIdx.x == 0) {
            isrc0 = cuda_getI(lastV, sim_time, olen, tp, output_cap);
            isrc0 = cuda_getI(curV + inv[idx(threadIdx.x, 0, n)] * isrc0, sim_time, olen, tp, output_cap);
        }
        __syncthreads();
        if(threadIdx.x < n) {
            curV += inv[idx(threadIdx.x, 0, n)] * isrc0;
            lastI = cap_g * (curV - lastV) - lastI;
        }
        if(threadIdx.x < m && lastV < 0.1 * VDD && curV >= 0.1 * VDD) {
            port_slew = lerp(sim_time - deltaT, sim_time, lastV, curV, 0.1 * VDD);
        }
        if(threadIdx.x < m && lastV < 0.5 * VDD && curV >= 0.5 * VDD && !reach50) {
            reach50 = true;
            atomicAdd(&reach50_count, 1);
        }
        if(threadIdx.x < m && lastV < 0.9 * VDD && curV >= 0.9 * VDD && !reached_vt) {
            port_slew = lerp(sim_time - deltaT, sim_time, lastV, curV, 0.9 * VDD) - port_slew;
            atomicAdd(&vt_reach_count, 1);
            reached_vt = true;
        }
        //if(threadIdx.x < m && lastV < 0.5 * VDD && curV >= 0.5 * VDD && net_id == 25340 && osignal == 1)
        //    printf("PASS0 threadIdx.x=%d  sim_time=%.4f\n", threadIdx.x, sim_time);
        if(threadIdx.x < n) lastV = curV;
        __syncthreads();
        if(threadIdx.x < m && reach50 && reach50_count * 2 <= m) early = true;

        if(vt_reach_count == m) break;
    }
    int offset = (blockIdx.x + stage_offset) * max_port_num;
    if(1 <= threadIdx.x && threadIdx.x < m) {
        if(EVALUATE == 0) atomicMax64(inslew + gate_ipin_acc_num[gate_num] * osignal + gate_ipin_acc_num[
            RC_port_gate_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x], port_slew);
        double *cap1_table = RC_ccs_cap + ((RC_port_offset + threadIdx.x) * 4 + osignal * 2) * TABLE_LEN;
        double *cap2_table = RC_ccs_cap + ((RC_port_offset + threadIdx.x) * 4 + osignal * 2 + 1) * TABLE_LEN;
        int cap1_idx = 0, cap2_idx = 0;
        double slew1 = port_slew, slew2 = port_slew, cap1_coeff = 0, cap2_coeff = 0;
        if(slew1 < 0.2 * CAP1_SLEWS[0]) slew1 = 0.2 * CAP1_SLEWS[0];
        if(slew1 > 1.1 * CAP1_SLEWS[TABLE_LEN - 1]) slew1 = 1.1 * CAP1_SLEWS[TABLE_LEN - 1];
        if(slew2 < 0.2 * CAP2_SLEWS[0]) slew2 = 0.2 * CAP2_SLEWS[0];
        if(slew2 > 1.1 * CAP2_SLEWS[TABLE_LEN - 1]) slew2 = 1.1 * CAP2_SLEWS[TABLE_LEN - 1];
        for(int i = TABLE_LEN - 2; i >= 0; i--)
            if(CAP1_SLEWS[i] < slew1) {cap1_idx = i; break; }
        for(int i = TABLE_LEN - 2; i >= 0; i--)
            if(CAP2_SLEWS[i] < slew2) {cap2_idx = i; break; }
        cap1_coeff = (slew1 - CAP1_SLEWS[cap1_idx]) / (CAP1_SLEWS[cap1_idx + 1] - CAP1_SLEWS[cap1_idx]);
        cap2_coeff = (slew2 - CAP2_SLEWS[cap2_idx]) / (CAP2_SLEWS[cap2_idx + 1] - CAP2_SLEWS[cap2_idx]);
        stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3] = 
            cap1_table[cap1_idx] * (1 - cap1_coeff) + cap1_table[cap1_idx + 1] * cap1_coeff;
        stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 1] = 
            cap2_table[cap2_idx] * (1 - cap2_coeff) + cap2_table[cap2_idx + 1] * cap2_coeff;
        stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 2] = 
            cap2_table[cap2_idx] * (1 - cap2_coeff) + cap2_table[cap2_idx + 1] * cap2_coeff;
        /*if(early) {
            stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 1] = 
            stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 2];
        } else {
            stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3 + 1] = 
            stage_port_cap[(stage_port_acc_num[stage_id] + threadIdx.x) * 3];
        }*/
    }
}
__global__ void pass1(int stage_offset) {
    extern __shared__ double shared[];

    int stage_id = sorted_stage_id[stage_offset + blockIdx.x];//for sorted
    //int stage_id = stage_offset + blockIdx.x;
    int net_id = stage_net_id[stage_id], gate_id = stage_driver_gate_id[stage_id], arc_id = stage_arc_id[stage_id];
    int ipin = stage_driver_ipin[stage_id], isignal = stage_signal[stage_id] / 2, osignal = stage_signal[stage_id] % 2;
    int RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    int olen = ccs_acc_olen[arc_id + 1] - ccs_acc_olen[arc_id];
    double *initial_tp = ccs_tp + ccs_acc_tlen[arc_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
    double *I_src = shared, *output_cap = I_src + n, *tp = output_cap + olen + 2;
    double cap_g;
    __shared__ double ref_time;

    if(threadIdx.x < n) cap_g = stage_cap_g[stage_node_acc_num[stage_id] + threadIdx.x];

    if(threadIdx.x == 0) output_cap[0] = ccs_output_cap[ccs_acc_olen[arc_id]] * 0.2;
    if(1 <= threadIdx.x && threadIdx.x <= olen) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + threadIdx.x - 1];
    if(threadIdx.x == olen + 1) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + olen - 1] * 1.1;

    assert(blockDim.x >= olen + 2);
    
    assert(blockDim.x >= n);
    {
        int ilen = ccs_acc_ilen[arc_id + 1] - ccs_acc_ilen[arc_id];
        __shared__ int index;
        __shared__ double coeff;
        if(threadIdx.x == 0) {
            double slew = inslew[gate_ipin_acc_num[gate_num] * isignal + gate_ipin_acc_num[gate_id] + ipin];
            
            double *input_slew = ccs_input_slew + ccs_acc_ilen[arc_id], *refer_time = ccs_refer_time + ccs_acc_ilen[arc_id];
            if(slew < input_slew[0] * 0.2) slew = input_slew[0] * 0.2;
            if(slew > input_slew[ilen - 1] * 1.1) slew = input_slew[ilen - 1] * 1.1;
            index = 0;
            for(int i = ilen - 2; i >= 0; i--) if(input_slew[i] < slew) { index = i; break; }
            coeff = (slew - input_slew[index]) / (input_slew[index + 1] - input_slew[index]);
            ref_time = refer_time[index] * (1 - coeff) + refer_time[index + 1] * coeff;
        }
        __syncthreads();
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN * olen; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN / olen, a = temp / olen % VPLEN, b = temp % olen;
            tp[idx3D(t, a, b + 1, VPLEN, olen + 2)] = initial_tp[idx3D(t, index, b, ilen, olen) * VPLEN + a] * (1 - coeff)
                + initial_tp[idx3D(t, index + 1, b, ilen, olen) * VPLEN + a] * coeff;
        }
        __syncthreads();
        olen += 2;
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN, a = temp % VPLEN, pos = idx3D(t, a, 0, VPLEN, olen);
            tp[pos] = tp[pos + 1] - (tp[pos + 2] - tp[pos + 1]) / (output_cap[2] - output_cap[1]) * (output_cap[1] - output_cap[0]);
            pos = idx3D(t, a, olen - 2, VPLEN, olen);
            tp[pos + 1] = tp[pos] + (tp[pos] - tp[pos - 1]) / (output_cap[olen - 2] - output_cap[olen - 3]) * (output_cap[olen - 1] - output_cap[olen - 2]);

        }
        __syncthreads();
    }
    const double deltaT = DELTA_T;
    double lastI = 0, lastV = 0.01 * VDD, curV, sim_time = tp[idx3D(1, 0, 0, VPLEN, olen)], port_slew = -1, port_time = 0;
    __shared__ double isrc0, driver_output_time;
    __shared__ int reach90_count, reach50_count;
    bool reach50 = false, reach90 = false;
    int iter = 0;
    if(threadIdx.x == 0) reach90_count = reach50_count = 0;
    __syncthreads();
    while(1) {
        sim_time += deltaT;
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        __syncthreads();
        if(threadIdx.x < n) for(int i = curV = 0; i < n; i++) curV += inv[idx(threadIdx.x, i, n)] * I_src[i];
        if(threadIdx.x == 0) {
            isrc0 = cuda_getI(lastV, sim_time, olen, tp, output_cap);
            isrc0 = cuda_getI(curV + inv[idx(threadIdx.x, 0, n)] * isrc0, sim_time, olen, tp, output_cap);
        }
        __syncthreads();
        if(threadIdx.x < n) {
            curV += inv[idx(threadIdx.x, 0, n)] * isrc0;
            lastI = cap_g * (curV - lastV) - lastI;
        }
        if(threadIdx.x < m) {
            if(lastV < 0.1 * VDD && curV >= 0.1 * VDD) 
                port_slew = lerp(sim_time - deltaT, sim_time, lastV, curV, 0.1 * VDD);
            if(lastV < 0.5 * VDD && curV >= 0.5 * VDD && !reach50) {
                port_time = lerp(sim_time - deltaT, sim_time, lastV, curV, 0.5 * VDD);
                atomicAdd(&reach50_count, 1);
                reach50 = true;
            }
            
            if(lastV < 0.9 * VDD && curV >= 0.9 * VDD && !reach90) {
                port_slew = lerp(sim_time - deltaT, sim_time, lastV, curV, 0.9 * VDD) - port_slew;
                atomicAdd(&reach90_count, 1);
                reach90 = true;
            }

        }
        if(threadIdx.x < n) lastV = curV;
        if(threadIdx.x == 0 && curV >= 0.5 * VDD) driver_output_time = port_time;
        __syncthreads();
        if(reach90_count == m) break;
        if(iter == 0 && reach50_count * 2 >= m) {
            iter++;
            inv += (INV3 ? 2 : 1) * stage_node2_acc_num[stage_num];
            if(threadIdx.x < n) cap_g = stage_cap_g[(INV3 ? 2 : 1) * stage_node_acc_num[stage_num] + stage_node_acc_num[stage_id] + threadIdx.x];
            //inv += stage_node2_acc_num[stage_num] * 2;
            //if(threadIdx.x < n) cap_g = stage_cap_g[stage_node_acc_num[stage_num] * 2 + stage_node_acc_num[stage_id] + threadIdx.x];
        }
        /*if(iter == 0 && reach50_count * 4 >= m) {
            iter++;
            inv += stage_node2_acc_num[stage_num];
            if(threadIdx.x < n) cap_g = stage_cap_g[stage_node_acc_num[stage_num] + stage_node_acc_num[stage_id] + threadIdx.x];
        }
        if(iter == 1 && reach50_count * 4 >= 3 * m) {
            iter++;
            inv += stage_node2_acc_num[stage_num];
            if(threadIdx.x < n) cap_g = stage_cap_g[stage_node_acc_num[stage_num] * 2 + stage_node_acc_num[stage_id] + threadIdx.x];
        }*/
        __syncthreads();
    }
    if(1 <= threadIdx.x && threadIdx.x < m)
        atomicMax64(RC_port_delay + RC_port_acc_num[net_num] * osignal + RC_port_offset + threadIdx.x, port_time - driver_output_time);
    if(threadIdx.x == 0) {
        stage_delay[stage_port_acc_num[stage_id]] = port_time - ref_time;
    } else if(threadIdx.x < m) {
        stage_delay[stage_port_acc_num[stage_id] + threadIdx.x] = port_time - driver_output_time;
    }
}

__global__ void propagate(int stage_offset) {
    int stage_id = blockIdx.x + stage_offset, net_id = stage_net_id[stage_id], gate_id = stage_driver_gate_id[stage_id];
    int isignal = stage_signal[stage_id] / 2, osignal = stage_signal[stage_id] % 2, ipin = stage_driver_ipin[stage_id];
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    if(1 <= threadIdx.x && threadIdx.x < m) {
        int idx = gate_ipin_acc_num[gate_num] * osignal + gate_ipin_acc_num[
            RC_port_gate_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x];
        atomicMax64(gate_ipin_delay + idx, gate_ipin_delay[gate_ipin_acc_num[gate_num] * isignal + gate_ipin_acc_num[gate_id] + ipin]
            + stage_delay[stage_port_acc_num[stage_id]] + RC_port_delay[RC_port_acc_num[net_num] * osignal + RC_port_offset + threadIdx.x]);
    }
    if(net_type[net_id] == 2 && threadIdx.x == 0)

        atomicMax64(output_delay + net_num * osignal + net_id, gate_ipin_delay[gate_ipin_acc_num[gate_num] * isignal + gate_ipin_acc_num[gate_id] + ipin]
            + stage_delay[stage_port_acc_num[stage_id]] + RC_port_delay[RC_port_acc_num[net_num] * osignal + RC_port_offset + 1]);
}


__global__ void calc_delay_0_general(int stage_offset) {
    extern __shared__ double shared[];
    __shared__ int achieve_num;//, stage_id, net_id, gate_id, arc_id, ipin, isignal, osignal
    if(threadIdx.x == 0) achieve_num = 0;

    int stage_id = blockIdx.x + stage_offset;
    int net_id = stage_net_id[stage_id], gate_id = stage_driver_gate_id[stage_id], arc_id = stage_arc_id[stage_id];
    int ipin = stage_driver_ipin[stage_id], isignal = stage_signal[stage_id] / 2, osignal = stage_signal[stage_id] % 2;
    int RC_node_offset = RC_node_acc_num[net_id], n = RC_node_acc_num[net_id + 1] - RC_node_offset;
    int RC_port_offset = RC_port_acc_num[net_id], m = RC_port_acc_num[net_id + 1] - RC_port_offset;
    int olen = ccs_acc_olen[arc_id + 1] - ccs_acc_olen[arc_id];
    double *initial_tp = ccs_tp + ccs_acc_tlen[arc_id], *g = stage_g + stage_node2_acc_num[stage_id], *inv = stage_inv + stage_node2_acc_num[stage_id];
    double *I_src = shared, *output_cap = I_src + n, *tp = output_cap + olen + 2;
    double cap_g, slew = inslew[gate_ipin_acc_num[gate_num] * isignal + gate_ipin_acc_num[gate_id] + ipin];

    if(threadIdx.x == 0) output_cap[0] = ccs_output_cap[ccs_acc_olen[arc_id]] * 0.2;
    if(1 <= threadIdx.x && threadIdx.x <= olen) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + threadIdx.x - 1];
    if(threadIdx.x == olen + 1) output_cap[threadIdx.x] = ccs_output_cap[ccs_acc_olen[arc_id] + olen - 1] * 1.1;

    assert(blockDim.x >= n);
    {
        int ilen = ccs_acc_ilen[arc_id + 1] - ccs_acc_ilen[arc_id];
        __shared__ int index;
        __shared__ double coeff;
        if(threadIdx.x == 0) {
            index = 0;
            double *input_slew = ccs_input_slew + ccs_acc_ilen[arc_id];
            for(int i = 1; i < ilen - 1; i++) if(input_slew[i] < slew) index = i;
            coeff = (slew - input_slew[index]) / (input_slew[index + 1] - input_slew[index]);
        }
        __syncthreads();
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN * olen; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN / olen, a = temp / olen % VPLEN, b = temp % olen;
            tp[idx3D(t, a, b + 1, VPLEN, olen + 2)] = initial_tp[idx3D(t, index, b, ilen, olen) * VPLEN + a] * (1 - coeff)
                + initial_tp[idx3D(t, index + 1, b, ilen, olen) * VPLEN + a] * coeff;
        }
        __syncthreads();
        olen += 2;
        for(int i = 0; i * blockDim.x + threadIdx.x < 3 * VPLEN; i++) {
            int temp = i * blockDim.x + threadIdx.x, t = temp / VPLEN, a = temp % VPLEN, pos = idx3D(t, a, 0, VPLEN, olen);
            tp[pos] = tp[pos + 1] - (tp[pos + 2] - tp[pos + 1]) / (output_cap[2] - output_cap[1]) * (output_cap[1] - output_cap[0]);
            pos = idx3D(t, a, olen - 2, VPLEN, olen);
            tp[pos + 1] = tp[pos] + (tp[pos] - tp[pos - 1]) / (output_cap[olen - 2] - output_cap[olen - 3]) * (output_cap[olen - 1] - output_cap[olen - 2]);
        }
        __syncthreads();
    }

    const double deltaT = 1;// slew / 20;

    //initialize g & inv
    for(int i = 0; i * blockDim.x + threadIdx.x < n * n; i++)
        g[i * blockDim.x + threadIdx.x] = inv[i * blockDim.x + threadIdx.x] = 0;
    __syncthreads();
    if(threadIdx.x < n) cap_g = RC_node_cap[RC_node_offset + threadIdx.x];
    if(threadIdx.x < n) inv[idx(threadIdx.x, threadIdx.x, n)] = 1;
    __syncthreads();
    if(1 <= threadIdx.x && threadIdx.x < m) cap_g += cell_ipin_nldm_cap[cell_ipin_acc_num[cell_num] * osignal +
        cell_ipin_acc_num[RC_port_cell_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x]];
    
    if(threadIdx.x < n)
        g[idx(threadIdx.x, threadIdx.x, n)] += (cap_g *= 2 / deltaT);
    __syncthreads();
    if(threadIdx.x < n && RC_res[RC_node_offset + threadIdx.x] > 0) {
        int u = RC_res_node0[RC_node_offset + threadIdx.x], v = RC_res_node1[RC_node_offset + threadIdx.x];
        double conductance = 1.0 / RC_res[RC_node_offset + threadIdx.x];
        atomicAdd(g + idx(u, u, n), conductance);
        atomicAdd(g + idx(v, v, n), conductance);
        atomicAdd(g + idx(u, v, n), -conductance);
        atomicAdd(g + idx(v, u, n), -conductance);
    }
    __syncthreads();



    //compute inv from g
    /*if(threadIdx.x == 0 && stage_id == 6063) {
        printf("  GOLDEN  initial  G\n");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++)
                printf("%11.3f", g[idx(i, j, n)]);
            printf("\n");
        }
    }*/
    
    for(int i = 0; i < n; i++) {
        for(int temp = 0; temp * blockDim.x + threadIdx.x < n * (n - i - 1); temp++) {
            int j = (temp * blockDim.x + threadIdx.x) / n + i + 1, k = (temp * blockDim.x + threadIdx.x) % n;
            if(k > i) g[idx(j, k, n)] -= g[idx(j, i, n)] / g[idx(i, i, n)] * g[idx(i, k, n)];
            inv[idx(j, k, n)] -= g[idx(j, i, n)] / g[idx(i, i, n)] * inv[idx(i, k, n)];
        }
        __syncthreads();
        //if(i < threadIdx.x && threadIdx.x < n) g[idx(threadIdx.x, i, n)] = 0;
    }
    __syncthreads();

    for(int i = n - 1; i >= 0; i--) {
        for(int temp = 0; temp * blockDim.x + threadIdx.x < n * i; temp++) {
            int j = (temp * blockDim.x + threadIdx.x) / n, k = (temp * blockDim.x + threadIdx.x) % n;
            if(k > i) g[idx(j, k, n)] -= g[idx(j, i, n)] / g[idx(i, i, n)] * g[idx(i, k, n)];
            inv[idx(j, k, n)] -= g[idx(j, i, n)] / g[idx(i, i, n)] * inv[idx(i, k, n)];
        }
        __syncthreads();
        if(threadIdx.x < n) inv[idx(i, threadIdx.x, n)] /= g[idx(i, i, n)];
    }
    __syncthreads();
    

    double lastI = 0, lastV = 0.01 * VDD, curV, sim_time = tp[idx3D(1, 0, 0, VPLEN, olen)], port_slew = 0;
    __shared__ double isrc0;
    while(1) {
        sim_time += deltaT;
        if(threadIdx.x < n) I_src[threadIdx.x] = cap_g * lastV + lastI;
        __syncthreads();
        if(threadIdx.x < n) for(int i = curV = 0; i < n; i++) curV += inv[idx(threadIdx.x, i, n)] * I_src[i];
        if(threadIdx.x == 0) {
            isrc0 = cuda_getI(lastV, sim_time, olen, tp, output_cap);
            isrc0 = cuda_getI(curV + inv[idx(threadIdx.x, 0, n)] * isrc0, sim_time, olen, tp, output_cap);
        }
        __syncthreads();
        if(threadIdx.x < n) {
            curV += inv[idx(threadIdx.x, 0, n)] * isrc0;
            lastI = cap_g * (curV - lastV) - lastI;
        }
        if(threadIdx.x < m && lastV < 0.1 * VDD && curV >= 0.1 * VDD)
            port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.1 * VDD);
        if(threadIdx.x < m && lastV < 0.9 * VDD && curV >= 0.9 * VDD) {
            port_slew = lerp(sim_time, sim_time + deltaT, lastV, curV, 0.9 * VDD) - port_slew;
            atomicAdd(&achieve_num, 1);
        }
        lastV = curV;
        __syncthreads();
        if(achieve_num == m) break;
    }
    if(1 <= threadIdx.x && threadIdx.x < m) 
        atomicMax64(inslew + gate_ipin_acc_num[gate_num] * osignal + gate_ipin_acc_num[
            RC_port_gate_id[RC_port_offset + threadIdx.x]] + RC_port_pin_id[RC_port_offset + threadIdx.x], port_slew);
    int offset = (blockIdx.x + stage_offset) * max_port_num;
    if(1 <= threadIdx.x && threadIdx.x < m)
        if(fabs(port_slew - GOLDEN[offset + threadIdx.x]) > 1e-5) {
            printf("block %d   slew diff: %.10f\n", blockIdx.x, fabs(port_slew - GOLDEN[offset + threadIdx.x]));
        }
}

vector<double> operator + (vector<double> a, vector<double> b) {
    for(auto e : b) a.emplace_back(e);
    return a;
}
void design::update_timing(double primary_input_slew, bool useCPU) {
    std::cout << "\n    --------------------------------------" << std::endl;
    std::cout << "    update timing: " << (useCPU ? "CPU" : "GPU") << " mode" << std::endl;
    std::cout << "    --------------------------------------\n" << std::endl;



    double GPU_sta_graph_start_time = clock();
    //reduce_RC();
    init_database_GPU();
    init_graph_GPU();
    output_log("init_database_GPU + init_graph_GPU: " + to_string((clock() - GPU_sta_graph_start_time) / CLOCKS_PER_SEC));


    if(EVALUATE == 1) {
        prepare();
        read_pt_results();
        cerr << "[DONE] READ PT" << endl;
        read_sp_results();
        cerr << "[DONE] READ SP" << endl;
    }




    if(useCPU) {
        for(auto &net : nets) net.build_mx();
        for(auto net_id : input)
            for(int signal = 0; signal < 2; signal++)
                compute_delay_input(primary_input_slew, nets[net_id], signal);
        DEBUG_ID = 0;
        for(auto vec : topo) {
            for(int iter = 1; iter < sizes.size(); iter++) {
                for(auto gate_id : vec) {
                    auto &gate = gates[gate_id];
                    auto &cell = CCS::cells[gate.cell_id];
                    for(int arc_id = 0; arc_id < cell.get_arc_size(); arc_id++) {
                        int ipin = arc_id / cell.get_opin_size(), opin = arc_id % cell.get_opin_size();
                        for(auto &arc : cell.arcs[arc_id])
                            for(int s = 0; s < 4; s++) if(arc.signal[s]) {
                                int n = nets[gate.o_nets[opin]].n;
                                if(!(sizes[iter - 1] < n && n <= sizes[iter])) continue;                    
                                auto delay = compute_delay_internal(gate.slews[s / 2][ipin], nets[gate.o_nets[opin]], arc.current[s % 2], arc.nldm_slew[s % 2], s % 2, s / 2, ipin, gate_id);                          
                                gate.update_arc_delay(delay, arc_id, s);                                
                                DEBUG_ID++;
                            }
                    }
                }
            }
        }
    } else {

    
        max_port_num = max_RC_port_count;
        vector<cudaStream_t> streams(sizes.size());
        for(auto &stream : streams) cudaStreamCreate(&stream);

        double gpu_input_net_time = clock();
        print_GPU_mem();
        {
            int *input_node2_acc_num_cpu = new int[input_num + 1](), *input_node2_acc_num;
            vector<int> RC_size_count(max_RC_node_count + 1, 0);
            for(int i = 0; i < input_num; i++) {
                int net_id = input_nets_cpu[i];
                input_node2_acc_num_cpu[i + 1] = input_node2_acc_num_cpu[i] + (nets[net_id].n + 1) * (nets[net_id].n + 1);
                RC_size_count[nets[net_id].n]++;
            }
            for(int i = max_RC_node_count; i >= 1; i--) RC_size_count[i - 1] += RC_size_count[i];
            cudaMalloc(&input_g, 2 * sizeof(double) * input_node2_acc_num_cpu[input_num]);
            cudaMalloc(&input_inv, 2 * sizeof(double) * input_node2_acc_num_cpu[input_num]);
            cudaMalloc(&input_cap_g, 2 * sizeof(double) * input_node2_acc_num_cpu[input_num]);
            cudaMalloc(&input_node2_acc_num, sizeof(int) * (input_num + 1));
            cudaMemcpy(input_node2_acc_num, input_node2_acc_num_cpu, sizeof(int) * (input_num + 1), cudaMemcpyHostToDevice);
            int *input_idx2net_cpu = new int[input_node2_acc_num_cpu[input_num]](), *input_idx2net;
            for(int i = 0; i < input_num; i++) {
                int n = nets[input_nets_cpu[i]].n + 1;
                for(int j = 0; j < n * n; j++) input_idx2net_cpu[input_node2_acc_num_cpu[i] + j] = i;
            }
            cudaMalloc(&input_idx2net, sizeof(int) * input_node2_acc_num_cpu[input_num]);
            cudaMemcpy(input_idx2net, input_idx2net_cpu, sizeof(int) * input_node2_acc_num_cpu[input_num], cudaMemcpyHostToDevice);
            build_input_mat<<<input_num * 2, max_RC_node_count + 1>>> (input_node2_acc_num, input_num);

            for(int i = 0; i <= max_RC_node_count; i++) if(RC_size_count[i] > 0)
                build_input_inv<<<BLOCK_NUM(2 * input_node2_acc_num_cpu[RC_size_count[i]]), THREAD_NUM>>>
                    (input_node2_acc_num, input_idx2net, input_node2_acc_num_cpu[RC_size_count[i]], input_node2_acc_num_cpu[input_num], i, 0);
            for(int i = max_RC_node_count; i >= 0; i--) if(RC_size_count[i] > 0)
                build_input_inv<<<BLOCK_NUM(2 * input_node2_acc_num_cpu[RC_size_count[i]]), THREAD_NUM>>>
                    (input_node2_acc_num, input_idx2net, input_node2_acc_num_cpu[RC_size_count[i]], input_node2_acc_num_cpu[input_num], i, 1);
            build_input_inv<<<BLOCK_NUM(2 * input_node2_acc_num_cpu[input_num]), THREAD_NUM>>>
                (input_node2_acc_num, input_idx2net, input_node2_acc_num_cpu[input_num], input_node2_acc_num_cpu[input_num], 0, 2);

            calc_delay_input_new<<<input_num, max_RC_node_count + 1, (max_RC_node_count + 1) * sizeof(double)>>> (input_node2_acc_num, input_node2_acc_num_cpu[input_num], 20, 0);
            calc_delay_input_new<<<input_num, max_RC_node_count + 1, (max_RC_node_count + 1) * sizeof(double)>>> (input_node2_acc_num, input_node2_acc_num_cpu[input_num], 20, 1);
        }
        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);
        output_log("GPU input net time: " + to_string((clock() - gpu_input_net_time) / CLOCKS_PER_SEC));


        GPU_sta_graph_start_time = clock();
        build_stage_matA<<<stage_num, max_RC_node_count>>> (0);
        for(int i = 0; i < max_RC_port_count; i++) build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], i, 0, 0);
        for(int i = max_RC_port_count - 1; i >= 0; i--) build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], i, 1, 0);
        build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], 0, 2, 0);
        build_stage_invBC<<<BLOCK_NUM(stage_portinode_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_portinode_acc_num_cpu[stage_num], 0, 0);
        build_stage_invBC<<<BLOCK_NUM(stage_portinode_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_portinode_acc_num_cpu[stage_num], 1, 0);
        build_stage_invD<<<BLOCK_NUM(stage_inode2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_inode2_acc_num_cpu[stage_num], 0);//rely on build_stage_invBC(0)

        cudaDeviceSynchronize();    
        assert(cudaGetLastError() == cudaSuccess);

        for(int i = 0; i < topo.size(); i++) {
            int size_sum = 0, size_max = 32;
            for(int j = 1; j < sizes.size(); j++) size_sum += size_num[i][j], size_max = max(size_max, size_max_node_num[i][j]);
            if(size_sum < 300)
                pass0<<<size_sum, size_max, (size_max + 430) * sizeof(double)>>> (size_offset[i][1]);
            else 
                for(int j = 1; j < sizes.size(); j++) if(size_num[i][j] > 0) {
                    pass0<<<size_num[i][j], max(32, size_max_node_num[i][j]), (size_max_node_num[i][j] + 430) * sizeof(double), streams[j]>>> (size_offset[i][j]);
                }
            cudaDeviceSynchronize();
        }
        build_stage_matA<<<stage_num, max_RC_node_count>>> (1);
        for(int i = 0; i < max_RC_port_count; i++) build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], i, 0, 1);
        for(int i = max_RC_port_count - 1; i >= 0; i--) build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], i, 1, 1);
        build_stage_invA<<<BLOCK_NUM(stage_port2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_port2_acc_num_cpu[stage_num], 0, 2, 1);

        build_stage_invBC<<<BLOCK_NUM(stage_portinode_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_portinode_acc_num_cpu[stage_num], 0, 1);
        build_stage_invBC<<<BLOCK_NUM(stage_portinode_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_portinode_acc_num_cpu[stage_num], 1, 1);

        build_stage_invD<<<BLOCK_NUM(stage_inode2_acc_num_cpu[stage_num]), THREAD_NUM>>> (stage_inode2_acc_num_cpu[stage_num], 1);//rely on build_stage_invBC(0)
        {
            int *sorted_stage_id_cpu = new int[stage_num];
            for(int i = 0; i < stage_num; i++) sorted_stage_id_cpu[i] = i;
            sort(sorted_stage_id_cpu, sorted_stage_id_cpu + stage_num, [&] (int l, int r) {
                int net_l = stage_net_id_cpu[l], net_r = stage_net_id_cpu[r];
                return nets[net_l].n < nets[net_r].n;
            });
            cudaMalloc(&sorted_stage_id, stage_num * sizeof(int));
            cudaMemcpy(sorted_stage_id, sorted_stage_id_cpu, sizeof(int) * stage_num, cudaMemcpyHostToDevice);
            vector<int> group_size = {32, 64, 128, 256, max_RC_node_count};
            vector<int> group_cnt = {0, 0, 0, 0, 0};
            for(int i = 0; i < stage_num; i++)
                for(int j = 0; j < group_size.size(); j++)
                    if(nets[stage_net_id_cpu[i]].n <= group_size[j]) group_cnt[j]++;
            for(int i = 0; i < group_size.size(); i++) {
                int offset = (i > 0 ? group_cnt[i - 1] : 0);
                if(group_cnt[i] <= offset) continue;
                pass1<<<group_cnt[i] - offset, group_size[i], (group_size[i] + 430) * sizeof(double)>>> (offset);
                cudaDeviceSynchronize();
            }
        }

        cudaDeviceSynchronize();


        assert(cudaGetLastError() == cudaSuccess);
        for(int i = 0; i < topo.size(); i++) {
            int size_sum = 0;
            for(int j = 1; j < sizes.size(); j++) size_sum += size_num[i][j];
            propagate<<<size_sum, max_RC_port_count>>> (size_offset[i][1]);
        }
        cudaDeviceSynchronize();
        double *output_delay_cpu = new double[net_num * 2];
        cudaMemcpy(output_delay_cpu, output_delay, sizeof(double) * net_num * 2, cudaMemcpyDeviceToHost);
        for(int i = 0; i < net_num; i++) if(net_type_cpu[i] == 2) {
            po_at[0][nets[i].net_name] = output_delay_cpu[i];
            po_at[1][nets[i].net_name] = output_delay_cpu[i + net_num];
        }
        output_log("GPU STA time: " + to_string((clock() - GPU_sta_graph_start_time) / CLOCKS_PER_SEC));

        auto eval = [&] (vector<double> &pt, vector<double> &my, vector<double> &sp, string info) {
            int success_count = 0, tot = sp.size();//sp.size();
            double sp_my = 0, sp_pt = 0, sp_my_abs = 0, sp_pt_abs = 0;
            for(int i = 0; i < tot; i++) if(sp[i] > 0) {
                success_count++;
                const double eps = 0.5;
                if(fabs(my[i] - sp[i]) >= eps) {
                    sp_my += fabs(my[i] - sp[i]) / sp[i];
                    sp_my_abs += fabs(my[i] - sp[i]);
                }
                if(fabs(pt[i] - sp[i]) >= eps) {
                    sp_pt += fabs(pt[i] - sp[i]) / sp[i];
                    sp_pt_abs += fabs(pt[i] - sp[i]);
                }
            }
            sp_my = sp_my / success_count * 100;
            sp_pt = sp_pt / success_count * 100;
            sp_my_abs /= success_count;
            sp_pt_abs /= success_count;

            printf("\n------------------------\n");
            cerr << "    " << info << endl;
            printf("MY-SP%20.2fps%20.2f%%\n", sp_my_abs, sp_my);
            printf("PT-SP%20.2fps%20.2f%%\n", sp_pt_abs, sp_pt);
            printf("SP failed: %d out of %d\n", tot - success_count, tot);
            printf("------------------------\n\n");
        };
        if(EVALUATE) {
            double *stage_delay_cpu = new double[stage_port_acc_num_cpu[stage_num]]();
            cudaMemcpy(stage_delay_cpu, stage_delay, sizeof(double) * stage_port_acc_num_cpu[stage_num], cudaMemcpyDeviceToHost);
            my_cell_fall = my_cell_rise = vector<double> (arc2idx_cell.size(), 0);
            my_net_fall = my_net_rise = vector<double> (arc2index_net.size(), 0);
            for(int i = 0; i < stage_num; i++) {
                int gate_id = stage_driver_gate_id_cpu[i], net_id = stage_net_id_cpu[i];
                int ipin = stage_driver_ipin_cpu[i], opin = nets[net_id].gates[0].second;
                int port_num = stage_port_acc_num_cpu[i + 1] - stage_port_acc_num_cpu[i];
                for(int j = 0; j < port_num; j++) {
                    double val = stage_delay_cpu[stage_port_acc_num_cpu[i] + j];
                    if(j == 0) {
                        auto &d = (stage_signal_cpu[i] % 2 ? my_cell_rise : my_cell_fall)[arc2idx_cell[make_tuple(gate_id, ipin, opin)]];
                        d = max(d, val);
                    } else {
                        auto &d = (stage_signal_cpu[i] % 2 ? my_net_rise : my_net_fall)[arc2index_net[make_pair(net_id, j)]];
                        d = max(d, val);
                    }
                }
            }

            eval(pt_cell_fall, my_cell_fall, sp_cell_fall, "cell_fall");
            eval(pt_cell_rise, my_cell_rise, sp_cell_rise, "cell_rise");
            eval(pt_net_fall, my_net_fall, sp_net_fall, "net_fall");
            eval(pt_net_rise, my_net_rise, sp_net_rise, "net_rise");
            vector<double> pt_all = pt_cell_fall + pt_cell_rise + pt_net_fall + pt_net_rise;
            vector<double> my_all = my_cell_fall + my_cell_rise + my_net_fall + my_net_rise;
            vector<double> sp_all = sp_cell_fall + sp_cell_rise + sp_net_fall + sp_net_rise;
            cerr << pt_all.size() << ' ' << my_all.size() << ' ' << sp_all.size() << endl;
            eval(pt_all, my_all, sp_all, "all");
        }

        
}


    if(useCPU) {

        for(auto vec : topo)
            for(auto gate_id : vec) {
                auto &gate = gates[gate_id];
                gate.update_output_at();
                for(auto net_id : gate.o_nets) {
                    auto &net = nets[net_id];
                    int opin = net.gates[0].second;
                    for(int i = 1; i < net.gates.size(); i++) {
                        auto &next_gate = gates[net.gates[i].first];
                        int ipin = net.gates[i].second;
                        for(int s = 0; s < 2; s++)
                            next_gate.update_input_at(gate.o_ats[s][opin] + next_gate.net_delays[s][ipin], ipin, s);
                    }
                    if(net.net_type == OUTPUT) 
                        for(int s = 0; s < 2; s++)
                            po_at[s][net.net_name] = gate.o_ats[s][opin] + po_net_delay[s][net.net_name];
                }
            }
    }
    


    bool REPORT_OUTPUT = true;
    if(REPORT_OUTPUT) {
        const int WIDTH = 15;
        cout << "\n " << setw(2 * WIDTH) << setfill('-') << " Output AT Report " << setw(WIDTH) << "-" << setfill(' ') << endl;
        cout << setw(WIDTH) << "Pin" << setw(WIDTH) << "Fall AT" << setw(WIDTH) << "Rise AT" << endl;
        for(auto net : nets) if(net.net_type == OUTPUT) {
            cout << setw(WIDTH) << net.net_name << setw(WIDTH) << po_at[0][net.net_name] << setw(WIDTH) << po_at[1][net.net_name] << endl;
        }
    }
}





}