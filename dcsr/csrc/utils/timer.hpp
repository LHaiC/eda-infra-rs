// timer.hpp
#pragma once
#include <chrono>
#include <iostream>

#define TIMER_SCOPE(name) \
    TimerScope timer_scope(name)

#define TIMER_START(var_name) \
    auto var_name = std::chrono::high_resolution_clock::now()

#define TIMER_END(start_var, name) \
    do { \
        auto end_var = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration<double, std::milli>(end_var - start_var).count(); \
        std::cout << name << ": " << duration << " ms" << std::endl; \
    } while(0)

#define TIMER_END_SILENT(start_var) \
    std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_var).count()

class TimerScope {
private:
    std::string _name;
    std::chrono::high_resolution_clock::time_point _start;

public:
    TimerScope(const std::string& name) : _name(name) {
        _start = std::chrono::high_resolution_clock::now();
    }

    ~TimerScope() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - _start).count();
        std::cout << _name << ": " << duration << " ms" << std::endl;
    }
};