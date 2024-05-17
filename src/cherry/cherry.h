#ifndef CHERRY_H
#define CHERRY_H

#include <math.h>
#include <time.h>

typedef struct CherryTimer {
    unsigned long long startTime;  
    unsigned long long endTime;

    double elapsedTime;  
} CherryTimer;

void cherry_start_timer(CherryTimer* timer);
void cherry_stop_timer(CherryTimer* timer);

#define CHERRY_START_RECORD(LABEL) static CherryTimer timer ## LABEL; cherry_start_timer(&timer ## LABEL);
#define CHERRY_STOP_RECORD(LABEL) cherry_stop_timer(&timer ## LABEL);
#define CHERRY_GET_ELAPSED_TIME(LABEL) (timer ## LABEL).elapsedTime

#endif