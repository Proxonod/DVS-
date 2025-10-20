#pragma once

#include <cstdint>
#include <vector>

#include "filter.hpp"

namespace dvs {

class RefractoryFilter : public Filter {
public:
    explicit RefractoryFilter(int64_t refractory_us = 500);

    void reset(int width, int height) override;
    void process(EventBuffer& events, PipelineState& state) override;

    [[nodiscard]] int64_t refractoryUs() const { return refractory_us_; }
    void setRefractoryUs(int64_t value);

private:
    int64_t refractory_us_;
    int width_{0};
    int height_{0};
    std::vector<int64_t> last_times_;
};

}  // namespace dvs
