#include <iostream>
#include <atomic>



  namespace ASC_HPC{
class Lock {
private:
    std::atomic<bool> locked = false; 

public:
    void lock() {
        bool expected = false;
        while (!locked.compare_exchange_strong(expected, true))
        {
            expected = false;
          
        }
    }

    void unlock() {
        locked = false;
    }
};}