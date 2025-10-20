#pragma once

#include <memory>

#include "event.hpp"
#include "overlay.hpp"

namespace dvs {

struct PipelineState {
    EventBuffer events;
    OverlayMap overlays;
};

class Filter {
public:
    virtual ~Filter() = default;
    virtual void reset(int width, int height) = 0;
    virtual void process(EventBuffer& events, PipelineState& state) = 0;
};

using FilterPtr = std::unique_ptr<Filter>;

}  // namespace dvs
