#include "cherry.h"

unsigned long long cherry_get_millis_internal(){
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC, &spec);
    return (unsigned long long)(spec.tv_sec) * 1000 + (spec.tv_nsec / 1000000);
}

void cherry_start_timer(CherryTimer* timer) {
    timer->startTime = cherry_get_millis_internal();
}

void cherry_stop_timer(CherryTimer* timer) {
    timer->endTime = cherry_get_millis_internal();
    timer->elapsedTime = (double)(timer->endTime - timer->startTime); 
}
