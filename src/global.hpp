#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <sstream>
#include <mutex>
#include <shared_mutex>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <dirent.h>
#include <vector>
#include <cstring>
#include <string_view>
#include <memory>
#include <map>
#include <future>
#include <atomic>
#include <list>
#include <forward_list>
#include <unordered_map>
#include <set>
#include <stack>
#include <queue>
#include <deque>
#include <tuple>
#include <unordered_set>
#include <numeric>
#include <iterator>
#include <functional>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <cassert>
#include <random>
#include <regex>
#include <ratio>
#include <optional>
#include <unistd.h>

#define myfloat double

#define VDD 0.7

template<class T>
std::vector<T> operator + (std::vector<T> a, const std::vector<T> &b) {
    for(int i = 0; i < a.size(); i++) a[i] += b[i];
    return a;
}

template<class T>
std::vector<T> operator - (std::vector<T> a, const std::vector<T> &b) {
    for(int i = 0; i < a.size(); i++) a[i] -= b[i];
    return a;
}

template<class T>
std::vector<myfloat> operator * (std::vector<myfloat> vec, T k) {
    for(auto &e : vec) e *= k;
    return vec;
}

int find(const std::vector<std::string> &tokens, std::string str, int l, int r = -1) {
    if(r == -1) r = tokens.size();
    for(int i = l; i < r; i++) if(tokens[i] == str) return i;
    return r;
}


std::vector<std::string> tokenize(std::string path, std::string dels, std::string exps) {
  
  using namespace std::literals::string_literals;

  std::ifstream ifs(path, std::ios::ate);

  if(!ifs.good()) {
    //throw std::invalid_argument("failed to open the file '"s + path.c_str() + '\'');
    return {};
  }
  
  // Read the file to a local buffer.

  double _t = clock();
  size_t fsize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buffer(fsize + 1);
  ifs.read(buffer.data(), fsize);
  buffer[fsize] = 0;
  
  // Mart out the comment
  for(size_t i=0; i<fsize; ++i) {

    // Block comment
    if(buffer[i] == '/' && buffer[i+1] == '*') {
      buffer[i] = buffer[i+1] = ' ';
      for(i=i+2; i<fsize; buffer[i++]=' ') {
        if(buffer[i] == '*' && buffer[i+1] == '/') {
          buffer[i] = buffer[i+1] = ' ';
          i = i+1;
          break;
        }
      }
    }
    
    // Line comment
    if(buffer[i] == '/' && buffer[i+1] == '/') {
      buffer[i] = buffer[i+1] = ' ';
      for(i=i+2; i<fsize; ++i) {
        if(buffer[i] == '\n' || buffer[i] == '\r') {
          break;
        }
        else buffer[i] = ' ';
      }
    }
    
    // Pond comment
    if(buffer[i] == '#') {
      buffer[i] = ' ';
      for(i=i+1; i<fsize; ++i) {
        if(buffer[i] == '\n' || buffer[i] == '\r') {
          break;
        }
        else buffer[i] = ' ';
      }
    }
  }

  //std::cerr << "read & remove comment in tokenize: " << (clock() - _t) / CLOCKS_PER_SEC << std::endl;
  //std::cout << std::string_view(buffer.data()) << std::endl;

  // Parse the token.
  std::string token;
  std::vector<std::string> tokens;
    //myfloat start_time = clock();
  for(size_t i=0; i<fsize; ++i) {

    auto c = buffer[i];
    bool is_del = (dels.find(c) != std::string::npos);

    if(is_del || std::isspace(c)) {
      if(!token.empty()) {                            // Add the current token.
        tokens.push_back(std::move(token));
        token.clear();
      }
      if(is_del && exps.find(c) != std::string::npos) {
        token.push_back(c);
        tokens.push_back(std::move(token));
      }
    } else {
      token.push_back(c);  // Add the char to the current token.
    }
  }
    //std::cerr << "Timed: " << (clock() - start_time) / CLOCKS_PER_SEC << std::endl;

  if(!token.empty()) {
    tokens.push_back(std::move(token));
  }

  return tokens;
}


// my matrix definition
struct mat {
	int n, m;
    std::vector<std::vector<myfloat>> val;
    //type = 0: 0 matrix; type = 1: identity matrix
    mat(int _n, int _m, int type = 0) : n(_n), m(_m) {
        val = std::vector<std::vector<myfloat>> (n, std::vector<myfloat> (m, 0));
        if(type) {
          assert(n == m);
          for(int i = 0; i < n; i++) val[i][i] = 1;
        }
    }
	mat() {}

    //adding conduntance g between i and j
    void addG(int i, int j, myfloat g) {
        assert(i != j);
        val[i][i] += g;
        val[j][j] += g;
        val[i][j] -= g;
        val[j][i] -= g;
    }
    //This is the case when x is connected to ground
    void addG(int x, myfloat g) {
        val[x][x] += g;
    }
    void print(std::string info = "") {
        std::cerr << "	---matrix---" << info << "---\n";
        int n = val.size();
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                std::cerr << std::setw(9) << std::setprecision(7) << (fabs(val[i][j]) < 1e-8 ? 0 : val[i][j]) << (" \n"[j + 1 == n]);
        std::cerr << "	----END----\n\n";
    }

    mat operator * (mat rhs) {
		assert(m == rhs.n);
        mat ans(n, rhs.m);
        for(int k = 0; k < m; k++)
            for(int i = 0; i < n; i++)
                for(int j = 0; j < rhs.m; j++)
                    ans.val[i][j] += val[i][k] * rhs.val[k][j];
        return ans;
    }
	mat operator + (mat rhs) {
		assert(n == rhs.n && m == rhs.m);
		mat ans(n, m);
		for(int i = 0; i < n; i++)
			for(int j = 0; j < m; j++) 
				ans.val[i][j] = val[i][j] + rhs.val[i][j];
		return ans;
	}	
	mat operator - (mat rhs) {
		assert(n == rhs.n && m == rhs.m);
		mat ans(n, m);
		for(int i = 0; i < n; i++)
			for(int j = 0; j < m; j++) 
				ans.val[i][j] = val[i][j] - rhs.val[i][j];
		return ans;
	}

    //compute inverse matrix
    mat inv() {
		assert(n == m);
        mat ans(n, n, 1), temp = *this;
        for(int i = 0; i < n; i++)
            for(int j = i + 1; j < n; j++) {
                myfloat coef = temp.val[j][i] / temp.val[i][i];
                for(int k = 0; k < n; k++)
                    temp.val[j][k] -= coef * temp.val[i][k], ans.val[j][k] -= coef * ans.val[i][k];
            }
        for(int i = n - 1; i >= 0; i--) {
            for(int j = 0; j < i; j++) {
                myfloat coef = temp.val[j][i] / temp.val[i][i];
                for(int k = 0; k < n; k++)
                    temp.val[j][k] -= coef * temp.val[i][k], ans.val[j][k] -= coef * ans.val[i][k];
            }
            for(int j = 0; j < n; j++) ans.val[i][j] /= temp.val[i][i];
        }
        if(0) {//verify inver matrix calculation
            const myfloat EPS = 1e-1;
            mat isI = *this * ans;
            for(int i = 0; i < n; i++)
                for(int j = 0; j < n; j++)
                    if((i != j && fabs(isI.val[i][j]) > EPS) || (i == j && fabs(isI.val[i][j] - 1) > EPS)) {
                        std::cerr << "Incorrect Inverse Matrix " << std::setprecision(20) << isI.val[i][j] << std::endl;
                        exit(-1);
                    }
        }
        return ans;
    }

	mat sub(int a, int b, int lena, int lenb) {
		mat ans(lena, lenb);
		for(int i = 0; i < lena; i++)
			for(int j = 0; j < lenb; j++)
				ans.val[i][j] = val[a + i][b + j];
		return ans;
	}	
	void sub(int a, int b, mat x) {
		for(int i = 0; i < x.n; i++)
			for(int j = 0; j < x.m; j++)
				val[a + i][b + j] = x.val[i][j];
	}
};

template<class T>
std::pair<int, T> get_index_coeff(const std::vector<T> &vec, T val) {
//std::pair<int, double> get_index_coeff(const std::vector<double> &vec, double val) {
	if(!(val >= vec[0] && val <= vec.back())) {
		std::cerr << val << " in ";
		for(auto e : vec) std::cerr << e << ' ';
		std::cerr << std::endl;
	}
    assert(val >= vec[0] && val <= vec.back());
    int index = 0;
    for(int i = 1; i < vec.size(); i++) 
        if(vec[i - 1] < val && val <= vec[i]) { index = i - 1; break; }
    return std::make_pair(index, (val - vec[index]) / (vec[index + 1] - vec[index]));
}

template<class T>
std::pair<int, T> get_index_coeff_margin(const std::vector<T> &vec, T val) {
    assert(val >= vec[0] * 0.2 && val <= vec.back() * 1.1);
    int index = 0;
    for(int i = vec.size() - 2; i >= 0; i--)
      if(vec[i] < val) {index = i; break; }
    //for(int i = 1; i < vec.size(); i++) //if(vec[i - 1] < val) index = i - 1;
    //    if(vec[i - 1] < val && val <= vec[i]) { index = i - 1; break; }
    return std::make_pair(index, (val - vec[index]) / (vec[index + 1] - vec[index]));
}

template<class T>
myfloat make_inbound(const std::vector<T> &vec, T val) {
	return std::max(vec[0], std::min(vec.back(), val));
}
template<class T>
myfloat make_inbound_margin(const std::vector<T> &vec, T val) {
	return std::max(myfloat(0.2) * vec[0], std::min(myfloat(1.1) * vec.back(), val));
}

template<class T>
T interpolate1D(const std::vector<T> &vec, std::pair<int, T> index_coeff) {
    return vec[index_coeff.first] * (1 - index_coeff.second) + vec[index_coeff.first + 1] * index_coeff.second;
}

void output_log(std::string info) {
  int n = info.size();
  const int margin = 2;
  std::cout << "\n";
  std::cout << "    " << std::setw(n + margin * 2) << std::setfill('-') << "-" << std::setfill(' ') << "\n";
  std::cout << "    " << std::setw(n + margin) << info << "\n";
  std::cout << "    " << std::setw(n + margin * 2) << std::setfill('-') << "-" << std::setfill(' ') << std::endl;
}