#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "filter.hpp"

namespace dvs {

class Pipeline {
public:
    Pipeline();

    void reset(int width, int height);
    void addFilter(FilterPtr filter);
    void clearFilters();

    [[nodiscard]] PipelineState processEvents(const EventBuffer& events);
    [[nodiscard]] std::vector<uint8_t> render(const PipelineState& state) const;

    void setPositiveColour(const std::array<uint8_t, 3>& colour);
    void setNegativeColour(const std::array<uint8_t, 3>& colour);

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

private:
    std::vector<FilterPtr> filters_;
    int width_{0};
    int height_{0};
    std::array<uint8_t, 3> pos_colour_;
    std::array<uint8_t, 3> neg_colour_;
};

std::array<uint8_t, 3> hexToRgb(const std::string& hex);

}  // namespace dvs
