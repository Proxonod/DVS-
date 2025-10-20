#include "dvs/pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace dvs {

namespace {
constexpr std::array<uint8_t, 3> kDefaultPosColour{0x00, 0xFF, 0xAA};
constexpr std::array<uint8_t, 3> kDefaultNegColour{0xFF, 0x33, 0x66};
}

Pipeline::Pipeline()
    : pos_colour_(kDefaultPosColour),
      neg_colour_(kDefaultNegColour) {}

void Pipeline::reset(int width, int height) {
    width_ = width;
    height_ = height;
    for (auto& filter : filters_) {
        filter->reset(width_, height_);
    }
}

void Pipeline::addFilter(FilterPtr filter) {
    if (!filter) {
        return;
    }
    filter->reset(width_, height_);
    filters_.emplace_back(std::move(filter));
}

void Pipeline::clearFilters() {
    filters_.clear();
}

PipelineState Pipeline::processEvents(const EventBuffer& events) {
    PipelineState state;
    state.events = events;
    for (auto& filter : filters_) {
        filter->process(state.events, state);
    }
    return state;
}

std::vector<uint8_t> Pipeline::render(const PipelineState& state) const {
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Pipeline dimensions are not initialised");
    }
    const auto& events = state.events;
    std::vector<uint8_t> frame(static_cast<size_t>(width_) * height_ * 3, 0);
    if (events.empty()) {
        return frame;
    }

    auto addColour = [](uint8_t* pixel, const std::array<uint8_t, 3>& colour) {
        for (int i = 0; i < 3; ++i) {
            int value = static_cast<int>(pixel[i]) + static_cast<int>(colour[i]);
            pixel[i] = static_cast<uint8_t>(std::min(value, 255));
        }
    };

    for (const auto& ev : events) {
        if (ev.x < 0 || ev.y < 0 || ev.x >= width_ || ev.y >= height_) {
            continue;
        }
        const size_t idx = (static_cast<size_t>(ev.y) * width_ + ev.x) * 3;
        auto* pixel = frame.data() + idx;
        if (ev.p) {
            addColour(pixel, pos_colour_);
        } else {
            addColour(pixel, neg_colour_);
        }
    }
    return frame;
}

void Pipeline::setPositiveColour(const std::array<uint8_t, 3>& colour) {
    pos_colour_ = colour;
}

void Pipeline::setNegativeColour(const std::array<uint8_t, 3>& colour) {
    neg_colour_ = colour;
}

std::array<uint8_t, 3> hexToRgb(const std::string& hex) {
    if (hex.size() != 7 || hex[0] != '#') {
        throw std::invalid_argument("Expected colour in format #RRGGBB");
    }
    auto hexToComponent = [](char high, char low) -> uint8_t {
        auto decode = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            throw std::invalid_argument("Invalid hex digit");
        };
        return static_cast<uint8_t>((decode(high) << 4) | decode(low));
    };
    return {hexToComponent(hex[1], hex[2]),
            hexToComponent(hex[3], hex[4]),
            hexToComponent(hex[5], hex[6])};
}

}  // namespace dvs
