#pragma once

#include <cstdint>
#include <vector>

namespace dvs {

struct Event {
    int32_t x{0};
    int32_t y{0};
    int64_t t{0};
    int8_t p{0};
};

using EventBuffer = std::vector<Event>;

}  // namespace dvs
