
#include "utils.cuh"

#ifdef HAS_PROFILER
StopWatchInterface *copy_host_to_device_timer = NULL;
StopWatchInterface *copy_device_to_host_timer = NULL;
StopWatchInterface *copy_device_to_device_timer = NULL;
StopWatchInterface *init_kernel_timer = NULL;
StopWatchInterface *apply_kernel_timer = NULL;
StopWatchInterface *measure_kernel_timer = NULL;

void init_timers()
{
    sdkCreateTimer(&copy_host_to_device_timer);
    sdkCreateTimer(&copy_device_to_host_timer);
    sdkCreateTimer(&copy_device_to_device_timer);
    sdkCreateTimer(&init_kernel_timer);
    sdkCreateTimer(&apply_kernel_timer);
    sdkCreateTimer(&measure_kernel_timer);
}
#endif
