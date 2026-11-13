#pragma once

// PC button glyphs: replaces the game's Wii U button-icon textures, found by
// the hash of their decoded pixels, with a generated key-cap image labelled
// with whatever PC input is bound to that pad button. Nothing is read from
// or shipped with the game; the 5x7 font below is drawn for this file.
//
// Table file (NWIIU_GLYPHS, else ./glyphs.ini):  <hex hash> = <PadButton>
// Hashes come from the file names NWIIU_TEXTURE_DUMP writes.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nwii::runtime {

class PcGlyphs {
public:
    using Labeler = std::function<std::string(const std::string& button)>;

    PcGlyphs() { load(); }

    void set_labeler(Labeler labeler) { labeler_ = std::move(labeler); }

    bool enabled() const { return !table_.empty(); }

    void apply(uint64_t hash, uint32_t width, uint32_t height,
               std::vector<uint8_t>& pixels) const {
        const auto found = table_.find(hash);
        if (found == table_.end() || width == 0 || height == 0) {
            return;
        }
        std::string label = labeler_ ? labeler_(found->second) : found->second;
        if (label.empty()) {
            label = found->second;
        }
        draw_keycap(label, width, height, pixels);
    }

    // Exposed for tests and for previewing labels.
    static void draw_keycap(const std::string& text, uint32_t width,
                            uint32_t height, std::vector<uint8_t>& pixels) {
        pixels.assign(static_cast<size_t>(width) * height * 4, 0);
        const float w = static_cast<float>(width);
        const float h = static_cast<float>(height);
        // Key cap: rounded rectangle, light rim, dark face.
        const float margin = std::max(1.f, std::min(w, h) * 0.04f);
        const float radius = std::min(w, h) * 0.22f;
        const float rim = std::max(1.f, std::min(w, h) * 0.07f);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                const float px = static_cast<float>(x) + 0.5f;
                const float py = static_cast<float>(y) + 0.5f;
                const float d = rounded_rect_distance(
                    px, py, margin, margin, w - margin, h - margin, radius);
                if (d > 0.5f) {
                    continue;
                }
                const float coverage = std::clamp(0.5f - d, 0.f, 1.f);
                uint8_t* p = &pixels[(static_cast<size_t>(y) * width + x) * 4];
                const bool on_rim = d > -rim;
                const uint8_t shade = on_rim ? 225 : 32;
                p[0] = p[1] = p[2] = shade;
                p[3] = static_cast<uint8_t>(coverage * (on_rim ? 255 : 215));
            }
        }
        // Text, scaled to fit inside the rim.
        std::string upper;
        for (char c : text) {
            upper.push_back(
                static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
        }
        if (upper.empty()) {
            return;
        }
        const float inner_w = w - 2 * (margin + rim) - 2;
        const float inner_h = h - 2 * (margin + rim) - 2;
        const float text_units = static_cast<float>(upper.size()) * 6 - 1;
        const float scale = std::max(
            1.f, std::floor(std::min(inner_w / text_units, inner_h / 7.f)));
        // A long label in a small icon may still overflow at scale 1; it is
        // clipped rather than shrunk below one pixel per font dot.
        const float text_w = text_units * scale;
        const float text_h = 7 * scale;
        const int x0 = static_cast<int>(std::round((w - text_w) / 2));
        const int y0 = static_cast<int>(std::round((h - text_h) / 2));
        for (size_t i = 0; i < upper.size(); ++i) {
            const auto& rows = glyph(upper[i]);
            for (int row = 0; row < 7; ++row) {
                for (int col = 0; col < 5; ++col) {
                    if ((rows[row] >> (4 - col) & 1) == 0) {
                        continue;
                    }
                    const int gx = x0 + static_cast<int>(
                                            (static_cast<float>(i) * 6 + col) *
                                            scale);
                    const int gy = y0 + static_cast<int>(row * scale);
                    fill(pixels, width, height, gx, gy,
                         static_cast<int>(scale), static_cast<int>(scale));
                }
            }
        }
    }

private:
    std::unordered_map<uint64_t, std::string> table_;
    Labeler labeler_;

    static float rounded_rect_distance(float px, float py, float x0, float y0,
                                       float x1, float y1, float r) {
        const float cx = std::clamp(px, x0 + r, x1 - r);
        const float cy = std::clamp(py, y0 + r, y1 - r);
        const float dx = px - cx;
        const float dy = py - cy;
        return std::sqrt(dx * dx + dy * dy) - r;
    }

    static void fill(std::vector<uint8_t>& pixels, uint32_t width,
                     uint32_t height, int x, int y, int w, int h) {
        for (int yy = std::max(0, y);
             yy < std::min(static_cast<int>(height), y + h); ++yy) {
            for (int xx = std::max(0, x);
                 xx < std::min(static_cast<int>(width), x + w); ++xx) {
                uint8_t* p =
                    &pixels[(static_cast<size_t>(yy) * width + xx) * 4];
                p[0] = p[1] = p[2] = 245;
                p[3] = 255;
            }
        }
    }

    // 5x7 font, one byte per row, bit 4 = leftmost column.
    static const std::array<uint8_t, 7>& glyph(char c) {
        static const std::unordered_map<char, std::array<uint8_t, 7>> kFont{
            {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
            {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
            {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
            {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
            {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
            {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
            {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
            {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
            {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
            {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
            {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
            {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
            {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}},
            {'N', {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11}},
            {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
            {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
            {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
            {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
            {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}},
            {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
            {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
            {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}},
            {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
            {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}},
            {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
            {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
            {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
            {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
            {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
            {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
            {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
            {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
            {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
            {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
            {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
            {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
            {'+', {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00}},
            {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
            {'^', {0x04, 0x0E, 0x15, 0x04, 0x04, 0x04, 0x04}}, // up arrow
            {'v', {0x04, 0x04, 0x04, 0x04, 0x15, 0x0E, 0x04}}, // down arrow
            {'<', {0x00, 0x04, 0x08, 0x1F, 0x08, 0x04, 0x00}}, // left arrow
            {'>', {0x00, 0x04, 0x02, 0x1F, 0x02, 0x04, 0x00}}, // right arrow
            {' ', {0, 0, 0, 0, 0, 0, 0}},
        };
        static const std::array<uint8_t, 7> kUnknown{0x0E, 0x11, 0x01, 0x02,
                                                     0x04, 0x00, 0x04};
        const auto found = kFont.find(c);
        return found != kFont.end() ? found->second : kUnknown;
    }

    void load() {
        const char* env = std::getenv("NWIIU_GLYPHS");
        const std::string path = env != nullptr ? env : "glyphs.ini";
        std::ifstream file(path);
        if (!file) {
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (const auto hash = line.find_first_of("#;");
                hash != std::string::npos) {
                line.resize(hash);
            }
            const auto equals = line.find('=');
            if (equals == std::string::npos) {
                continue;
            }
            const std::string key = line.substr(0, equals);
            std::string value = line.substr(equals + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t\r") + 1);
            char* end = nullptr;
            const uint64_t texture = std::strtoull(key.c_str(), &end, 16);
            if (end != key.c_str() && !value.empty()) {
                table_[texture] = value;
            }
        }
        std::fprintf(stderr, "glyphs: %zu replacements from %s\n",
                     table_.size(), path.c_str());
    }
};

} // namespace nwii::runtime
