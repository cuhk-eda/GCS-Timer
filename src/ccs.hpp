#pragma once
#include "global.hpp"

namespace CCS {

    
using namespace std;

const vector<myfloat> VP = vector<myfloat> ({0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97}) * VDD;
const myfloat deltaV = VP[0];

struct ccs_table {

    ccs_table() {}
    ccs_table(const vector<string> &tokens, int l, int r);

    void prebuild();
    myfloat build(myfloat in_slew);
    myfloat getI(myfloat v, myfloat t);

    int id;
    vector<vector<vector<myfloat>>> V_times[3];
    vector<myfloat> refer_time, input_slew, output_cap;
    vector<vector<vector<pair<myfloat, myfloat>>>> output_current; //(input_slew x output_cap) -> vector of pair (current, time)
    vector<vector<myfloat>> tp[3];// v -> [output_cap] = time
};

struct table2D {

    table2D() {}
    table2D(const vector<string> &tokens, int l, int r);

    myfloat lookup(myfloat slew, myfloat cap);
    vector<myfloat> lookup_cap(myfloat cap) {
        vector<myfloat> ans(input_slew.size());
        for(int i = 0; i < ans.size(); i++) ans[i] = lookup(input_slew[i], cap);
        return ans;
    }

    vector<myfloat> input_slew, output_cap;
    vector<vector<myfloat>> values;
};

struct arc {

    arc(const vector<string> &tokens, const vector<int> &match_pos, int l, int r);

    bool signal[4];
    ccs_table current[2];
    table2D nldm_delay[2], nldm_slew[2], ccs_cap[2][2];
};

struct cell {

    cell(const vector<string> &tokens, const vector<int> &match_pos, int l, int r);
    cell() {}

    int get_ipin_index(string pin_name);
    int get_opin_index(string pin_name);
    int get_arc_index(string ipin_name, string opin_name);
    int get_arc_index(int ipin_idx, int opin_idx);
    int get_ipin_size();
    int get_opin_size();
    int get_arc_size();


    string cell_name;
    vector<string> i_pin_name, o_pin_name;
    vector<myfloat> nldm_cap[2];
    vector<vector<arc>> arcs;
};
vector<cell> cells;

    
table2D::table2D(const vector<string> &tokens, int l, int r) {
    int cur = find(tokens, "index_1", l) + 2;
    while(tokens[cur] != ")") input_slew.emplace_back(stod(tokens[cur++]));
    cur = find(tokens, "index_2", cur) + 2;
    while(tokens[cur] != ")") output_cap.emplace_back(stod(tokens[cur++]));
    values = vector<vector<myfloat>> (input_slew.size(), vector<myfloat> (output_cap.size()));
    cur = find(tokens, "values", cur) + 2;
    for(int i = 0; i < input_slew.size(); i++)
        for(int j = 0; j < output_cap.size(); j++)
            values[i][j] = stod(tokens[cur++]);
    assert(tokens[cur] == ")");
}

ccs_table::ccs_table(const vector<string> &tokens, int l, int r) {
    set<myfloat> _refer_time, _input_slew, _output_cap;
    for(int cur = find(tokens, "vector", l, r); cur < r; cur = find(tokens, "vector", cur, r)) {
        _refer_time.insert(stod(tokens[cur = find(tokens, "reference_time", cur) + 1]));
        _input_slew.insert(stod(tokens[cur = find(tokens, "index_1", cur) + 2]));
        _output_cap.insert(stod(tokens[cur = find(tokens, "index_2", cur) + 2]));
    }
    for(auto e : _refer_time) refer_time.emplace_back(e);
    for(auto e : _input_slew) input_slew.emplace_back(e);
    for(auto e : _output_cap) output_cap.emplace_back(e);
    output_current = vector<vector<vector<pair<myfloat, myfloat>>>> (input_slew.size(), vector<vector<pair<myfloat, myfloat>>> (output_cap.size()));
    for(int cur = find(tokens, "vector", l, r); cur < r; cur = find(tokens, "vector", cur, r)) {
        myfloat s = stod(tokens[cur = find(tokens, "index_1", cur) + 2]);
        myfloat c = stod(tokens[cur = find(tokens, "index_2", cur) + 2]);
        int s_idx = find(input_slew.begin(), input_slew.end(), s) - input_slew.begin();
        int c_idx = find(output_cap.begin(), output_cap.end(), c) - output_cap.begin();
        assert(input_slew[s_idx] == s && output_cap[c_idx] == c);
        auto &t_i = output_current[s_idx][c_idx];
        cur = find(tokens, "index_3", cur) + 2;
        while(tokens[cur] != ")") t_i.emplace_back(make_pair(stod(tokens[cur++]), 0));
        cur = find(tokens, "values", cur) + 2;
        for(int i = 0; i < t_i.size(); i++) t_i[i].second = fabs(stod(tokens[cur++]));
    }
    prebuild();
}

arc::arc(const vector<string> &tokens, const vector<int> &match_pos, int l, int r) {
    signal[0] = signal[1] = signal[2] = signal[3] = 0;
    int cur = find(tokens, "timing_sense", l) + 1;
    if(tokens[cur] == "positive_unate" || tokens[cur] == "non_unate") signal[0] = signal[3] = 1;
    if(tokens[cur] == "negative_unate" || tokens[cur] == "non_unate") signal[1] = signal[2] = 1;
    for( ; cur < r; cur++) {
        if(tokens[cur].find("output_current_") == string::npos
            && tokens[cur].find("receiver_capacitance") == string::npos
            && tokens[cur].find("cell_") == string::npos
            && tokens[cur].find("_transition") == string::npos) continue;
        int end = match_pos[find(tokens, "{", cur)];
        if(tokens[cur] == "output_current_rise") current[1] = ccs_table(tokens, cur, end);
        if(tokens[cur] == "output_current_fall") current[0] = ccs_table(tokens, cur, end);
        if(tokens[cur] == "rise_transition") nldm_slew[1] = table2D(tokens, cur, end);
        if(tokens[cur] == "fall_transition") nldm_slew[0] = table2D(tokens, cur, end);
        if(tokens[cur] == "cell_rise") nldm_delay[1] = table2D(tokens, cur, end); 
        if(tokens[cur] == "cell_fall") nldm_delay[0] = table2D(tokens, cur, end);
        if(tokens[cur] == "receiver_capacitance1_rise") ccs_cap[1][0] = table2D(tokens, cur, end);
        if(tokens[cur] == "receiver_capacitance2_rise") ccs_cap[1][1] = table2D(tokens, cur, end);
        if(tokens[cur] == "receiver_capacitance1_fall") ccs_cap[0][0] = table2D(tokens, cur, end);
        if(tokens[cur] == "receiver_capacitance2_fall") ccs_cap[0][1] = table2D(tokens, cur, end);
        cur = end;
    }
}

cell::cell(const vector<string> &tokens, const vector<int> &match_pos, int l, int r) {

    cell_name = tokens[l + 2];
    vector<pair<int, int>> ipin_range, opin_range;


    for(int cur = find(tokens, "pin", l); cur < r; cur = find(tokens, "pin", cur)) {
        int end = match_pos[find(tokens, "{", cur)];
        auto direction = tokens[find(tokens, "direction", cur) + 1];
        assert(direction == "input" || direction == "output");
        (direction == "output" ? opin_range : ipin_range).emplace_back(cur, end);
        cur = end;
    }


    for(auto e : opin_range) o_pin_name.emplace_back(tokens[e.first + 2]);
    for(auto e : ipin_range) {
        i_pin_name.emplace_back(tokens[e.first + 2]);
        for(int i = e.first; i < e.second; i++) {
            if(tokens[i] == "fall_capacitance") nldm_cap[0].emplace_back(stod(tokens[i + 1]));
            if(tokens[i] == "rise_capacitance") nldm_cap[1].emplace_back(stod(tokens[i + 1]));
        }
    }
    arcs.resize(get_arc_size());

    for(auto e : opin_range)
        for(int cur = e.first; cur < e.second; cur = find(tokens, "timing", cur, e.second)) if(tokens[cur] == "timing") {
            auto end = match_pos[find(tokens, "{", cur)];
            auto arc_idx = get_arc_index(tokens[find(tokens, "related_pin", cur) + 1], tokens[e.first + 2]);            
            if(arc_idx != -1) arcs[arc_idx].emplace_back(arc(tokens, match_pos, cur, end));// discard output-to-output arcs for now
            cur = end;
        }
}

myfloat table2D::lookup(myfloat slew, myfloat cap) {
    auto s_idx_coe = get_index_coeff_margin(input_slew, slew = make_inbound_margin(input_slew, slew));
    auto c_idx_coe = get_index_coeff_margin(output_cap, cap = make_inbound_margin(output_cap, cap));
    //auto c_idx_coe = get_index_coeff_margin(output_cap, cap = make_inbound_margin(output_cap, cap));
    return values[s_idx_coe.first][c_idx_coe.first] * (1 - s_idx_coe.second) * (1 - c_idx_coe.second) + 
           values[s_idx_coe.first][c_idx_coe.first + 1] * (1 - s_idx_coe.second) * c_idx_coe.second + 
           values[s_idx_coe.first + 1][c_idx_coe.first] * s_idx_coe.second * (1 - c_idx_coe.second) + 
           values[s_idx_coe.first + 1][c_idx_coe.first + 1] * s_idx_coe.second * c_idx_coe.second;
}

int cell::get_ipin_size() {
    return i_pin_name.size();
}
int cell::get_opin_size() {
    return o_pin_name.size();
}
int cell::get_arc_size() {
    return get_ipin_size() * get_opin_size();
}

myfloat ccs_table::getI(myfloat v, myfloat t) {
    auto v_idx_coe = get_index_coeff(VP, v = make_inbound(VP, v));
    auto cap2time = tp[1][v_idx_coe.first] * (1 - v_idx_coe.second) + tp[1][v_idx_coe.first + 1] * v_idx_coe.second;
    auto t_idx_coe = get_index_coeff(cap2time, t = make_inbound(cap2time, t));
    myfloat deltaT0 = interpolate1D(tp[2][v_idx_coe.first], t_idx_coe) - interpolate1D(tp[0][v_idx_coe.first], t_idx_coe);
    myfloat deltaT1 = interpolate1D(tp[2][v_idx_coe.first + 1], t_idx_coe) - interpolate1D(tp[0][v_idx_coe.first + 1], t_idx_coe);
    auto ans = interpolate1D(output_cap, t_idx_coe) * 2 * deltaV * ((1 - v_idx_coe.second) / deltaT0 + v_idx_coe.second / deltaT1);
    return ans;
}

void ccs_table::prebuild() {
    for(int offset = 0; offset < 3; offset++) {
        vector<vector<vector<myfloat>>> temp_t(input_slew.size(), vector<vector<myfloat>> (output_cap.size()));
        for(int i = 0; i < input_slew.size(); i++)
            for(int j = 0; j < output_cap.size(); j++) {
                auto &t_i = output_current[i][j];
                vector<myfloat> q = {0};//charges
                for(int k = 1; k < t_i.size(); k++)
                    q.emplace_back(q.back() + (t_i[k].first - t_i[k - 1].first) * (t_i[k].second + t_i[k - 1].second) / 2);
                for(int k = 0; k < VP.size(); k++) {
                    myfloat target_q = (VP[k] + (offset - 1) * deltaV) * output_cap[j];
                    if(target_q < q.back()) {
                        auto idx = get_index_coeff(q, target_q).first;
                        myfloat coe = (t_i[idx + 1].second - t_i[idx].second) / (t_i[idx + 1].first - t_i[idx].first);
                        if(coe == 0) {
                            temp_t[i][j].emplace_back(t_i[idx].first + (target_q - q[idx]) / t_i[idx].second);
                        } else
                            temp_t[i][j].emplace_back(t_i[idx].first + (-t_i[idx].second + sqrt(t_i[idx].second * t_i[idx].second + 2 * coe * (target_q - q[idx]))) / coe);
                    } else {
                        temp_t[i][j].emplace_back(t_i.back().first);
                    }
                }
            }
        V_times[offset] = move(temp_t);
        tp[offset] = vector<vector<myfloat>> (VP.size(), vector<myfloat> (2 + output_cap.size()));
    }
    output_cap.resize(output_cap.size() + 1);
    for(int i = output_cap.size() - 1; i >= 1; i--) output_cap[i] = output_cap[i - 1];
    output_cap[0] = output_cap[1] * 0.2;
    output_cap.emplace_back(output_cap.back() * 1.1);
}

myfloat ccs_table::build(myfloat in_slew) {
    auto idx_coe = get_index_coeff_margin(input_slew, in_slew = make_inbound_margin(input_slew, in_slew));
    for(int offset = 0; offset < 3; offset++) {
        auto &temp_t = V_times[offset];
        for(int i = 0; i < VP.size(); i++) {
            for(int j = 1; j + 1 < output_cap.size(); j++) {
                tp[offset][i][j] = temp_t[idx_coe.first][j - 1][i] * (1 - idx_coe.second) + temp_t[idx_coe.first + 1][j - 1][i] * idx_coe.second;
                assert(!isnan(tp[offset][i][j]));
            }
            tp[offset][i][0] = tp[offset][i][1] - 
                (tp[offset][i][2] - tp[offset][i][1]) / (output_cap[2] - output_cap[1]) * (output_cap[1] - output_cap[0]);
            int last = tp[offset][i].size() - 2;
            tp[offset][i][last + 1] = tp[offset][i][last] + 
                (tp[offset][i][last] - tp[offset][i][last - 1]) / (output_cap[last] - output_cap[last - 1]) * (output_cap[last + 1] - output_cap[last]);
        }
    }
    return interpolate1D(refer_time, idx_coe);
}

int cell::get_ipin_index(string pin_name) {
    auto idx = find(i_pin_name.begin(), i_pin_name.end(), pin_name) - i_pin_name.begin();
    return idx < i_pin_name.size() ? idx : -1;
}
int cell::get_opin_index(string pin_name) {
    auto idx = find(o_pin_name.begin(), o_pin_name.end(), pin_name) - o_pin_name.begin();
    return idx < o_pin_name.size() ? idx : -1;
}
int cell::get_arc_index(int ipin_idx, int opin_idx) {
    return ipin_idx == -1 || opin_idx == -1 ? -1 : ipin_idx * int(o_pin_name.size()) + opin_idx;
}
int cell::get_arc_index(string ipin_name, string opin_name) {
    return get_arc_index(get_ipin_index(ipin_name), get_opin_index(opin_name));
}


int get_cell_id(string cell_name) {
    for(int cell_id = 0; cell_id < cells.size(); cell_id++)
        if(cells[cell_id].cell_name == cell_name) return cell_id;
    cerr << "Error in getting cell id: cell name " << cell_name << endl;
    return -1;
}

void read_lib(string filename) {

    double t = clock();
    auto tokens = tokenize(filename, "(),:;/#[]{}*\"\\", "(){}");

    auto match_pos = [&tokens] {
        vector<int> ans(tokens.size()), temp;
        for(int i = 0; i < tokens.size(); i++) {
            if(tokens[i] == "{") temp.emplace_back(i);
            if(tokens[i] == "}") ans[temp.back()] = i, temp.pop_back();
        }
        return move(ans);
    } ();// match_pos[i]=j indicates that the left bracket at tokens[i] is matched with the right bracket at tokens[j]



    vector<pair<int, int>> cell_token_range;
    for(int cur = 0; cur < tokens.size(); cur = find(tokens, "cell", cur)) if(tokens[cur] == "cell") {
        cell_token_range.emplace_back(cur, match_pos[find(tokens, "{", cur)]);
        cur = cell_token_range.back().second;
    }
    double _t = clock();
    for(auto range : cell_token_range)
        cells.emplace_back(cell(tokens, match_pos, range.first, range.second));

}

}
