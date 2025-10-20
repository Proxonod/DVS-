#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace dvs {

struct Overlay {
    int width{0};
    int height{0};
    std::vector<float> data;
};

using OverlayMap = std::unordered_map<std::string, Overlay>;

}  // namespace dvs
