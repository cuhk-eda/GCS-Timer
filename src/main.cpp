#include "gpu_timer.hpp"

void start() {
    std::cout << std::endl;
    std::cout << std::setw(30) << "* * * * * * * * *" << std::endl;
    std::cout << std::setw(30) << "*               *" << std::endl;
    std::cout << std::setw(30) << "*   GCS-Timer   *" << std::endl;
    std::cout << std::setw(30) << "*               *" << std::endl;
    std::cout << std::setw(30) << "* * * * * * * * *" << std::endl;
    std::cout << std::endl;
}

void read_lib() {
    double t = clock();
    
    std::vector<std::string> files = {
        "lib/asap7sc7p5t_INVBUF_RVT_TT_ccs_220122.lib",
        "lib/asap7sc7p5t_SIMPLE_RVT_TT_ccs_211120.lib",
        "lib/asap7sc7p5t_AO_RVT_TT_ccs_211120.lib", 
        "lib/asap7sc7p5t_OA_RVT_TT_ccs_211120.lib"
    };

    bool multithreaded = false;

    if(multithreaded) {
        std::vector<std::thread> threads;
        for(auto file : files) threads.emplace_back(std::thread(CCS::read_lib, file));
        for(auto &thread : threads) thread.join();
    } else 
        for(auto file : files) {
            CCS::read_lib(file);
        }
}

const std::string benchmark_path = "bm/";

int main(int argc, char* argv[]) {

    start();
    read_lib();

    std::string design_name(argv[1]);


    GPU_TIMER::design design(design_name, benchmark_path + design_name);
    double t = clock();
    design.update_timing(20, argc > 2 && std::string(argv[2]) == "-CPU");
    output_log("update_timing: " + std::to_string((clock() - t) / CLOCKS_PER_SEC) + "s");

    return 0;
}
