#include "dvs/refractory_filter.hpp"

#include <algorithm>
#include <limits>

namespace dvs {

RefractoryFilter::RefractoryFilter(int64_t refractory_us)
    : refractory_us_(refractory_us) {}

void RefractoryFilter::setRefractoryUs(int64_t value) {
    refractory_us_ = value;
}

void RefractoryFilter::reset(int width, int height) {
    width_ = width;
    height_ = height;
    const int64_t min_value = std::numeric_limits<int64_t>::min() / 2;
    last_times_.assign(static_cast<size_t>(width_) * height_, min_value);
}

void RefractoryFilter::process(EventBuffer& events, PipelineState&) {
    if (events.empty() || width_ <= 0 || height_ <= 0) {
        return;
    }
    EventBuffer filtered;
    filtered.reserve(events.size());
    const auto limit = static_cast<size_t>(width_) * height_;
    for (const auto& ev : events) {
        if (ev.x < 0 || ev.y < 0 || ev.x >= width_ || ev.y >= height_) {
            continue;
        }
        const size_t idx = static_cast<size_t>(ev.y) * width_ + ev.x;
        if (idx >= limit) {
            continue;
        }
        int64_t& last = last_times_[idx];
        if (ev.t - last >= refractory_us_) {
            filtered.push_back(ev);
            last = ev.t;
        }
    }
    events.swap(filtered);
}

}  // namespace dvs
