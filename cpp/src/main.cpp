#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "dvs/baf_filter.hpp"
#include "dvs/pipeline.hpp"
#include "dvs/refractory_filter.hpp"
#include "dvs/time_surface_filter.hpp"

using namespace std::chrono;

namespace dvs {

namespace {

bool parseInt(const std::string& token, int& value) {
    try {
        size_t idx = 0;
        const int parsed = std::stoi(token, &idx, 10);
        if (idx != token.size()) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseInt64(const std::string& token, int64_t& value) {
    try {
        size_t idx = 0;
        const long long parsed = std::stoll(token, &idx, 10);
        if (idx != token.size()) return false;
        value = static_cast<int64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parseEvents(const std::string& path, EventBuffer& events) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open events file: " << path << "\n";
        return false;
    }
    events.clear();
    events.reserve(1 << 16);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        std::string token;
        Event ev{};
        if (!std::getline(iss, token, ',')) return false;
        if (!parseInt(token, ev.x)) return false;
        if (!std::getline(iss, token, ',')) return false;
        if (!parseInt(token, ev.y)) return false;
        if (!std::getline(iss, token, ',')) return false;
        if (!parseInt64(token, ev.t)) return false;
        if (!std::getline(iss, token, ',')) return false;
        int polarity = 0;
        if (!parseInt(token, polarity)) return false;
        ev.p = static_cast<int8_t>(polarity != 0);
        events.push_back(ev);
    }
    return true;
}

bool writePpm(const std::string& path, int width, int height, const std::vector<uint8_t>& frame) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output frame: " << path << "\n";
        return false;
    }
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(frame.data()), static_cast<std::streamsize>(frame.size()));
    return true;
}

void printUsage() {
    std::cout << "Usage: dvs_cli <width> <height> <events.csv> <output.ppm>\n";
    std::cout << "CSV format: x,y,t,p with integer values (microsecond timestamps)." << std::endl;
}

}  // namespace

}  // namespace dvs

int main(int argc, char** argv) {
    using namespace dvs;
    if (argc < 5) {
        printUsage();
        return 1;
    }

    int width = 0;
    int height = 0;
    if (!parseInt(argv[1], width) || !parseInt(argv[2], height) || width <= 0 || height <= 0) {
        std::cerr << "Invalid sensor dimensions." << std::endl;
        return 1;
    }

    const std::string input_path = argv[3];
    const std::string output_path = argv[4];

    EventBuffer events;
    if (!parseEvents(input_path, events)) {
        std::cerr << "Failed to parse events file." << std::endl;
        return 1;
    }

    Pipeline pipeline;
    pipeline.reset(width, height);
    pipeline.addFilter(std::make_unique<RefractoryFilter>(500));
    pipeline.addFilter(std::make_unique<BackgroundActivityFilter>(10.0, 1, 0, 1));
    pipeline.addFilter(std::make_unique<TimeSurfaceFilter>(10.0, true));

    const auto start = high_resolution_clock::now();
    auto state = pipeline.processEvents(events);
    const auto frame = pipeline.render(state);
    const auto end = high_resolution_clock::now();
    const auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "Processed " << events.size() << " events in " << duration << "us" << std::endl;

    if (!writePpm(output_path, width, height, frame)) {
        return 1;
    }
    std::cout << "Frame written to " << output_path << std::endl;
    return 0;
}
