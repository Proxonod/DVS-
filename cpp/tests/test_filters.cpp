#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dvs/baf_filter.hpp"
#include "dvs/pipeline.hpp"
#include "dvs/refractory_filter.hpp"
#include "dvs/time_surface_filter.hpp"

using dvs::BackgroundActivityFilter;
using dvs::Event;
using dvs::EventBuffer;
using dvs::Pipeline;
using dvs::PipelineState;
using dvs::RefractoryFilter;
using dvs::TimeSurfaceFilter;

namespace {

EventBuffer makeEvents(const std::vector<std::pair<int, int>>& coords,
                       const std::vector<int64_t>& times,
                       const std::vector<int>& polarities) {
    EventBuffer events;
    const size_t n = coords.size();
    events.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Event ev{};
        ev.x = coords[i].first;
        ev.y = coords[i].second;
        ev.t = times[i];
        ev.p = static_cast<int8_t>(polarities[i]);
        events.push_back(ev);
    }
    return events;
}

void testRefractoryFilter() {
    RefractoryFilter filter(500);
    filter.reset(4, 4);
    auto events = makeEvents({{1, 1}, {1, 1}, {1, 1}}, {0, 100, 700}, {1, 1, 1});
    dvs::PipelineState state;
    filter.process(events, state);
    assert(events.size() == 2);
    assert(events[0].t == 0);
    assert(events[1].t == 700);
}

void testBackgroundActivityFilter() {
    BackgroundActivityFilter filter(20.0, 2, 0, 2);
    filter.reset(6, 6);
    auto events = makeEvents({{2, 2}, {1, 2}, {3, 2}}, {0, 2000, 5000}, {1, 1, 1});
    dvs::PipelineState state;
    filter.process(events, state);
    assert(events.size() == 1);
    assert(events[0].x == 3 && events[0].y == 2);
}

void testTimeSurfaceFilter() {
    TimeSurfaceFilter filter(10.0, true);
    filter.reset(3, 3);
    auto events1 = makeEvents({{1, 1}}, {0}, {1});
    dvs::PipelineState state1;
    filter.process(events1, state1);
    auto initial = state1.overlays["time_surface_pos"].data;

    auto events2 = makeEvents({{0, 0}}, {5000}, {1});
    dvs::PipelineState state2;
    filter.process(events2, state2);
    const auto& pos_surface = state2.overlays["time_surface_pos"].data;
    const float expected_decay = std::exp(-5000.0 / (10.0 * 1000.0));
    assert(std::abs(pos_surface[1 * 3 + 1] - initial[1 * 3 + 1] * expected_decay) < 1e-3);
    assert(std::abs(pos_surface[0 * 3 + 0] - 1.0f) < 1e-6);
}

void testPipelineColours() {
    Pipeline pipeline;
    pipeline.reset(4, 4);
    auto events = makeEvents({{0, 0}, {1, 1}, {2, 2}}, {0, 0, 0}, {1, 0, 1});
    auto state = pipeline.processEvents(events);
    auto frame = pipeline.render(state);
    const auto pos = dvs::hexToRgb("#00FFAA");
    const auto neg = dvs::hexToRgb("#FF3366");
    auto pixel = [&](int x, int y) {
        const size_t idx = (static_cast<size_t>(y) * 4 + x) * 3;
        return std::array<uint8_t, 3>{frame[idx], frame[idx + 1], frame[idx + 2]};
    };
    assert(pixel(0, 0) == pos);
    assert(pixel(1, 1) == neg);
    assert(pixel(2, 2) == pos);
}

}  // namespace

int main() {
    testRefractoryFilter();
    testBackgroundActivityFilter();
    testTimeSurfaceFilter();
    testPipelineColours();
    return 0;
}
