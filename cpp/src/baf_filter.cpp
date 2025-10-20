#include "dvs/baf_filter.hpp"

#include <algorithm>
#include <limits>

namespace dvs {

BackgroundActivityFilter::BackgroundActivityFilter(double window_ms,
                                                   int count_threshold,
                                                   int64_t refractory_us,
                                                   int spatial_radius)
    : window_us_(static_cast<int64_t>(window_ms * 1000.0)),
      count_threshold_(std::max(1, count_threshold)),
      refractory_us_(refractory_us),
      spatial_radius_(std::max(0, spatial_radius)) {}

void BackgroundActivityFilter::setWindowMs(double window_ms) {
    window_us_ = static_cast<int64_t>(window_ms * 1000.0);
}

void BackgroundActivityFilter::setCountThreshold(int threshold) {
    count_threshold_ = std::max(1, threshold);
}

void BackgroundActivityFilter::setRefractoryUs(int64_t value) {
    refractory_us_ = value;
}

void BackgroundActivityFilter::setSpatialRadius(int radius) {
    spatial_radius_ = std::max(0, radius);
}

void BackgroundActivityFilter::reset(int width, int height) {
    width_ = width;
    height_ = height;
    const int64_t safe_neg = -std::max(window_us_, refractory_us_) - 1;
    last_times_.assign(static_cast<size_t>(width_) * height_, safe_neg);
}

void BackgroundActivityFilter::process(EventBuffer& events, PipelineState&) {
    if (events.empty() || width_ <= 0 || height_ <= 0) {
        return;
    }
    EventBuffer filtered;
    filtered.reserve(events.size());
    const auto total = static_cast<size_t>(width_) * height_;
    for (const auto& ev : events) {
        if (ev.x < 0 || ev.y < 0 || ev.x >= width_ || ev.y >= height_) {
            continue;
        }
        const size_t idx = static_cast<size_t>(ev.y) * width_ + ev.x;
        if (idx >= total) {
            continue;
        }
        int64_t& last = last_times_[idx];
        const int64_t dt = ev.t - last;
        if (dt < refractory_us_) {
            last = ev.t;
            continue;
        }

        int x0 = std::max(0, ev.x - spatial_radius_);
        int y0 = std::max(0, ev.y - spatial_radius_);
        int x1 = std::min(width_ - 1, ev.x + spatial_radius_);
        int y1 = std::min(height_ - 1, ev.y + spatial_radius_);

        int count = 0;
        for (int yy = y0; yy <= y1 && count < count_threshold_; ++yy) {
            const size_t row = static_cast<size_t>(yy) * width_;
            for (int xx = x0; xx <= x1; ++xx) {
                const int64_t neighbour_dt = ev.t - last_times_[row + xx];
                if (neighbour_dt <= window_us_) {
                    ++count;
                    if (count >= count_threshold_) {
                        break;
                    }
                }
            }
        }
        if (count >= count_threshold_) {
            filtered.push_back(ev);
        }
        last = ev.t;
    }
    events.swap(filtered);
}

}  // namespace dvs
