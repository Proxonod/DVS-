#include "dvs/time_surface_filter.hpp"

#include <algorithm>
#include <cmath>

namespace dvs {

TimeSurfaceFilter::TimeSurfaceFilter(double tau_ms, bool polarity_separate)
    : tau_us_(tau_ms * 1000.0),
      polarity_separate_(polarity_separate) {}

void TimeSurfaceFilter::setTauMs(double tau_ms) {
    tau_us_ = tau_ms * 1000.0;
}

void TimeSurfaceFilter::setPolaritySeparate(bool separate) {
    polarity_separate_ = separate;
}

void TimeSurfaceFilter::reset(int width, int height) {
    width_ = width;
    height_ = height;
    surface_pos_.assign(static_cast<size_t>(width_) * height_, 0.0f);
    surface_neg_.assign(static_cast<size_t>(width_) * height_, 0.0f);
    last_timestamp_.reset();
}

float TimeSurfaceFilter::decayFactor(double dt_us) const {
    if (dt_us <= 0.0 || tau_us_ <= 0.0) {
        return 1.0f;
    }
    const double exponent = -dt_us / tau_us_;
    if (exponent < -50.0) {
        return 0.0f;
    }
    return static_cast<float>(std::exp(exponent));
}

void TimeSurfaceFilter::decaySurfaces(float decay) {
    if (decay == 1.0f) {
        return;
    }
    for (auto& v : surface_pos_) {
        v *= decay;
    }
    for (auto& v : surface_neg_) {
        v *= decay;
    }
}

void TimeSurfaceFilter::process(EventBuffer& events, PipelineState& state) {
    if (events.empty() || width_ <= 0 || height_ <= 0) {
        return;
    }
    const int64_t current = events.back().t;
    double dt = 0.0;
    if (last_timestamp_) {
        dt = static_cast<double>(current - *last_timestamp_);
        if (dt < 0.0) {
            dt = 0.0;
        }
    }
    last_timestamp_ = current;

    const float decay = decayFactor(dt);
    decaySurfaces(decay);

    const auto total = static_cast<size_t>(width_) * height_;
    for (const auto& ev : events) {
        if (ev.x < 0 || ev.y < 0 || ev.x >= width_ || ev.y >= height_) {
            continue;
        }
        const size_t idx = static_cast<size_t>(ev.y) * width_ + ev.x;
        if (idx >= total) {
            continue;
        }
        if (polarity_separate_) {
            if (ev.p) {
                surface_pos_[idx] = 1.0f;
            } else {
                surface_neg_[idx] = 1.0f;
            }
        } else {
            surface_pos_[idx] = 1.0f;
        }
    }

    Overlay pos_overlay{width_, height_, surface_pos_};
    state.overlays["time_surface_pos"] = pos_overlay;
    if (polarity_separate_) {
        Overlay neg_overlay{width_, height_, surface_neg_};
        state.overlays["time_surface_neg"] = neg_overlay;
    } else {
        state.overlays["time_surface_neg"] = pos_overlay;
    }
}

}  // namespace dvs
