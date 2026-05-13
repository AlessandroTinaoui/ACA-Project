#ifndef GEM5_STATS_H
#define GEM5_STATS_H

#ifndef KAN_ENABLE_GEM5_M5OPS
#define KAN_ENABLE_GEM5_M5OPS 0
#endif

#if KAN_ENABLE_GEM5_M5OPS
#include <gem5/m5ops.h>

static inline void kan_gem5_reset_stats(void) {
    m5_reset_stats(0, 0);
}

static inline void kan_gem5_dump_stats(void) {
    m5_dump_stats(0, 0);
}
#else
static inline void kan_gem5_reset_stats(void) {
}

static inline void kan_gem5_dump_stats(void) {
}
#endif

#endif
