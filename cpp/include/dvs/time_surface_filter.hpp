#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "filter.hpp"

namespace dvs {

class TimeSurfaceFilter : public Filter {
public:
    explicit TimeSurfaceFilter(double tau_ms = 50.0, bool polarity_separate = true);

    void reset(int width, int height) override;
    void process(EventBuffer& events, PipelineState& state) override;

    void setTauMs(double tau_ms);
    void setPolaritySeparate(bool separate);

private:
    float decayFactor(double dt_us) const;
    void decaySurfaces(float decay);

    double tau_us_;
    bool polarity_separate_;
    int width_{0};
    int height_{0};
    std::optional<int64_t> last_timestamp_;
    std::vector<float> surface_pos_;
    std::vector<float> surface_neg_;
};

}  // namespace dvs
