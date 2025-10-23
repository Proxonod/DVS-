#pragma once

#include <cstdint>
#include <vector>

#include "filter.hpp"

namespace dvs {

class BackgroundActivityFilter : public Filter {
public:
    BackgroundActivityFilter(double window_ms = 20.0,
                             int count_threshold = 2,
                             int64_t refractory_us = 1000,
                             int spatial_radius = 2);

    void reset(int width, int height) override;
    void process(EventBuffer& events, PipelineState& state) override;

    void setWindowMs(double window_ms);
    void setCountThreshold(int threshold);
    void setRefractoryUs(int64_t value);
    void setSpatialRadius(int radius);

private:
    int64_t window_us_;
    int count_threshold_;
    int64_t refractory_us_;
    int spatial_radius_;
    int width_{0};
    int height_{0};
    std::vector<int64_t> last_times_;
};

}  // namespace dvs
