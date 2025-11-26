#include <iostream>
#include <algorithm>

#include <taskmanager.hpp>

#include <timer.hpp>
#include <lock_guard.hpp>

#include <sstream>



using namespace ASC_HPC;
using std::cout, std::endl;




int main(){
    Lock lock1;
    StartWorkers(3);
  
    int cnt = 0;
    RunParallel(1000, [&cnt, &lock1] (int i, int size) {
        lock1.lock();
        cnt++;
        lock1.unlock();
        });

    cout<<cnt<<endl;

    StopWorkers();
    



}