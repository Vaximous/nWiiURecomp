#pragma once

// Keyboard, mouse and SDL-gamepad input mapped onto the Wii U GamePad
// (VPAD) the game reads. Bindings come from a small "key = Button" text
// file (NWIIU_CONTROLS, else ./controls.ini); without one the defaults
// below apply. The game itself also has a remappable Options > Controls
// screen, so these bind PC inputs to *pad* buttons, not to actions.

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace nwii::runtime {

class PcInput {
public:
    // VPAD button bits.
    static constexpr uint32_t kA = 0x8000, kB = 0x4000, kX = 0x2000,
                              kY = 0x1000, kLeft = 0x0800, kRight = 0x0400,
                              kUp = 0x0200, kDown = 0x0100, kZL = 0x0080,
                              kZR = 0x0040, kL = 0x0020, kR = 0x0010,
                              kPlus = 0x0008, kMinus = 0x0004,
                              kHome = 0x0002, kStickR = 0x00020000,
                              kStickL = 0x00040000;

    // Virtual "buttons" above the VPAD mask drive the left stick.
    enum Virtual : uint32_t {
        kMoveUp = 1u << 0,
        kMoveDown = 1u << 1,
        kMoveLeft = 1u << 2,
        kMoveRight = 1u << 3,
        kWalk = 1u << 4, // halves the left stick while held
    };

    struct Binding {
        enum class Source { key, mouse_button, wheel_up, wheel_down };
        Source source{};
        int code{};           // SDL scancode or mouse button index
        uint32_t buttons{};   // VPAD mask
        uint32_t virtuals{};  // Virtual mask
        std::string input;    // as written in the config, for glyph labels
    };

    struct State {
        uint32_t buttons{};
        float lx{}, ly{}, rx{}, ry{};
    };

    PcInput() {
        load();
        SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    }

    ~PcInput() {
        // The renderer's destructor may already have called SDL_Quit.
        if (gamepad_ != nullptr && SDL_WasInit(SDL_INIT_GAMEPAD) != 0) {
            SDL_CloseGamepad(gamepad_);
        }
    }

    PcInput(const PcInput&) = delete;
    PcInput& operator=(const PcInput&) = delete;

    // Feed every SDL event here before polling.
    void handle_event(SDL_Window* window, const SDL_Event& event) {
        switch (event.type) {
        case SDL_EVENT_MOUSE_MOTION:
            if (captured_) {
                mouse_dx_ += event.motion.xrel;
                mouse_dy_ += event.motion.yrel;
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (!captured_ && capture_mouse_) {
                // The click that captures the mouse is not passed on.
                set_capture(window, true);
                swallow_button_ = event.button.button;
                break;
            }
            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (swallow_button_ == event.button.button) {
                swallow_button_ = 0;
            }
            break;
        case SDL_EVENT_MOUSE_WHEEL:
            if (event.wheel.y > 0) {
                wheel_up_frames_ = kWheelHoldPolls;
            } else if (event.wheel.y < 0) {
                wheel_down_frames_ = kWheelHoldPolls;
            }
            break;
        case SDL_EVENT_KEY_DOWN:
            if (event.key.scancode == release_key_ && captured_) {
                set_capture(window, false);
            }
            break;
        case SDL_EVENT_WINDOW_FOCUS_LOST:
            set_capture(window, false);
            break;
        case SDL_EVENT_GAMEPAD_ADDED:
            if (gamepad_ == nullptr) {
                gamepad_ = SDL_OpenGamepad(event.gdevice.which);
            }
            break;
        case SDL_EVENT_GAMEPAD_REMOVED:
            if (gamepad_ != nullptr &&
                SDL_GetGamepadID(gamepad_) == event.gdevice.which) {
                SDL_CloseGamepad(gamepad_);
                gamepad_ = nullptr;
            }
            break;
        default:
            break;
        }
    }

    State poll() {
        State state;
        uint32_t virtuals = 0;
        const bool* keys = SDL_GetKeyboardState(nullptr);
        float mx = 0;
        float my = 0;
        const SDL_MouseButtonFlags mouse = SDL_GetMouseState(&mx, &my);
        for (const auto& binding : bindings_) {
            bool down = false;
            switch (binding.source) {
            case Binding::Source::key:
                down = keys[binding.code];
                break;
            case Binding::Source::mouse_button:
                down = captured_ && swallow_button_ != binding.code &&
                       (mouse & SDL_BUTTON_MASK(binding.code)) != 0;
                break;
            case Binding::Source::wheel_up:
                down = wheel_up_frames_ > 0;
                break;
            case Binding::Source::wheel_down:
                down = wheel_down_frames_ > 0;
                break;
            }
            if (down) {
                state.buttons |= binding.buttons;
                virtuals |= binding.virtuals;
            }
        }
        wheel_up_frames_ = std::max(0, wheel_up_frames_ - 1);
        wheel_down_frames_ = std::max(0, wheel_down_frames_ - 1);

        float lx = ((virtuals & kMoveRight) ? 1.f : 0.f) -
                   ((virtuals & kMoveLeft) ? 1.f : 0.f);
        float ly = ((virtuals & kMoveUp) ? 1.f : 0.f) -
                   ((virtuals & kMoveDown) ? 1.f : 0.f);
        if (lx != 0 && ly != 0) {
            lx *= 0.7071f;
            ly *= 0.7071f;
        }
        if (virtuals & kWalk) {
            lx *= 0.5f;
            ly *= 0.5f;
        }

        // Mouse look: the stick reports mouse *velocity*, so camera speed
        // (stick x rate x frame time) tracks hand movement at any frame
        // rate. Velocity is averaged over >= 1/60 s windows to smooth the
        // ~100 Hz poll.
        const uint64_t now = SDL_GetTicksNS();
        if (look_window_start_ == 0) {
            look_window_start_ = now;
        }
        const double elapsed = static_cast<double>(now - look_window_start_) /
                               1e9;
        if (elapsed >= 1.0 / 60.0) {
            const double scale = sensitivity_ / 1000.0 / elapsed;
            look_x_ = static_cast<float>(mouse_dx_ * scale);
            look_y_ = static_cast<float>(-mouse_dy_ * scale *
                                         (invert_y_ ? -1.0 : 1.0));
            mouse_dx_ = 0;
            mouse_dy_ = 0;
            look_window_start_ = now;
        }
        float rx = look_x_;
        float ry = look_y_;
        if (!captured_) {
            rx = ry = 0;
        }

        if (gamepad_ != nullptr) {
            poll_gamepad(state.buttons, lx, ly, rx, ry);
        }
        state.lx = std::clamp(lx, -1.f, 1.f);
        state.ly = std::clamp(ly, -1.f, 1.f);
        state.rx = std::clamp(rx, -1.f, 1.f);
        state.ry = std::clamp(ry, -1.f, 1.f);
        return state;
    }

    bool captured() const { return captured_; }

    // Short key-cap label for the first PC input bound to a pad button
    // name ("ZR" -> "LMB"). "LStick"/"RStick" name the sticks themselves.
    std::string label(const std::string& button) const {
        const auto name = lower(button);
        if (name == "lstick") {
            return "WASD";
        }
        if (name == "rstick") {
            return "MOUSE";
        }
        uint32_t buttons = 0;
        uint32_t virtuals = 0;
        if (!target(button, buttons, virtuals)) {
            return {};
        }
        for (const auto& binding : bindings_) {
            if ((binding.buttons & buttons) != 0 ||
                (binding.virtuals & virtuals) != 0) {
                return short_name(binding.input);
            }
        }
        return {};
    }

    static std::string short_name(const std::string& input) {
        static const std::array<std::pair<const char*, const char*>, 22>
            kShort{{{"mouse1", "LMB"},        {"mouse2", "RMB"},
                    {"mouse3", "MMB"},        {"mouse4", "M4"},
                    {"mouse5", "M5"},         {"wheelup", "WHL^"},
                    {"wheeldown", "WHLv"},    {"left shift", "SHIFT"},
                    {"right shift", "SHIFT"}, {"left ctrl", "CTRL"},
                    {"right ctrl", "CTRL"},   {"left alt", "ALT"},
                    {"right alt", "ALT"},     {"return", "ENTER"},
                    {"escape", "ESC"},        {"backspace", "BKSP"},
                    {"space", "SPACE"},       {"tab", "TAB"},
                    {"up", "^"},              {"down", "v"},
                    {"left", "<"},            {"right", ">"}}};
        const auto name = lower(input);
        for (const auto& [from, to] : kShort) {
            if (name == from) {
                return to;
            }
        }
        return input;
    }

private:
    static constexpr int kWheelHoldPolls = 12; // ~a game frame or two

    std::vector<Binding> bindings_;
    double sensitivity_{1.0};
    bool invert_y_{};
    bool capture_mouse_{true};
    SDL_Scancode release_key_{SDL_SCANCODE_F1};
    bool captured_{};
    int swallow_button_{};
    double mouse_dx_{};
    double mouse_dy_{};
    float look_x_{};
    float look_y_{};
    uint64_t look_window_start_{};
    int wheel_up_frames_{};
    int wheel_down_frames_{};
    SDL_Gamepad* gamepad_{};
    float deadzone_{0.15f};

    void set_capture(SDL_Window* window, bool capture) {
        if (captured_ == capture || window == nullptr) {
            return;
        }
        captured_ = capture;
        SDL_SetWindowRelativeMouseMode(window, capture);
        mouse_dx_ = mouse_dy_ = 0;
        look_x_ = look_y_ = 0;
    }

    float axis(SDL_GamepadAxis which) const {
        const float value =
            static_cast<float>(SDL_GetGamepadAxis(gamepad_, which)) / 32767.f;
        if (std::fabs(value) < deadzone_) {
            return 0;
        }
        const float sign = value < 0 ? -1.f : 1.f;
        return sign * (std::fabs(value) - deadzone_) / (1.f - deadzone_);
    }

    void poll_gamepad(uint32_t& buttons, float& lx, float& ly, float& rx,
                      float& ry) {
        // Positional layout: the bottom face button is the pad's B, as on
        // the Wii U Pro Controller.
        static constexpr std::array<std::pair<SDL_GamepadButton, uint32_t>, 15>
            kMap{{{SDL_GAMEPAD_BUTTON_SOUTH, kB},
                  {SDL_GAMEPAD_BUTTON_EAST, kA},
                  {SDL_GAMEPAD_BUTTON_WEST, kY},
                  {SDL_GAMEPAD_BUTTON_NORTH, kX},
                  {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, kL},
                  {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, kR},
                  {SDL_GAMEPAD_BUTTON_START, kPlus},
                  {SDL_GAMEPAD_BUTTON_BACK, kMinus},
                  {SDL_GAMEPAD_BUTTON_GUIDE, kHome},
                  {SDL_GAMEPAD_BUTTON_LEFT_STICK, kStickL},
                  {SDL_GAMEPAD_BUTTON_RIGHT_STICK, kStickR},
                  {SDL_GAMEPAD_BUTTON_DPAD_UP, kUp},
                  {SDL_GAMEPAD_BUTTON_DPAD_DOWN, kDown},
                  {SDL_GAMEPAD_BUTTON_DPAD_LEFT, kLeft},
                  {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, kRight}}};
        for (const auto& [button, bit] : kMap) {
            if (SDL_GetGamepadButton(gamepad_, button)) {
                buttons |= bit;
            }
        }
        if (axis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 0.4f) {
            buttons |= kZL;
        }
        if (axis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 0.4f) {
            buttons |= kZR;
        }
        lx += axis(SDL_GAMEPAD_AXIS_LEFTX);
        ly -= axis(SDL_GAMEPAD_AXIS_LEFTY);
        rx += axis(SDL_GAMEPAD_AXIS_RIGHTX);
        ry -= axis(SDL_GAMEPAD_AXIS_RIGHTY);
    }

    static std::string trim(std::string text) {
        const auto first = text.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return {};
        }
        const auto last = text.find_last_not_of(" \t\r\n");
        return text.substr(first, last - first + 1);
    }

    static std::string lower(std::string text) {
        for (auto& c : text) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return text;
    }

    // Returns false for an unknown name.
    static bool target(const std::string& name, uint32_t& buttons,
                       uint32_t& virtuals) {
        static const std::array<std::pair<const char*, uint32_t>, 17> kButtons{
            {{"a", kA}, {"b", kB}, {"x", kX}, {"y", kY}, {"l", kL},
             {"r", kR}, {"zl", kZL}, {"zr", kZR}, {"plus", kPlus},
             {"minus", kMinus}, {"home", kHome}, {"up", kUp},
             {"down", kDown}, {"left", kLeft}, {"right", kRight},
             {"stickl", kStickL}, {"stickr", kStickR}}};
        static const std::array<std::pair<const char*, uint32_t>, 5> kVirtual{
            {{"moveup", kMoveUp}, {"movedown", kMoveDown},
             {"moveleft", kMoveLeft}, {"moveright", kMoveRight},
             {"walk", kWalk}}};
        const auto key = lower(name);
        for (const auto& [n, bit] : kButtons) {
            if (key == n) {
                buttons |= bit;
                return true;
            }
        }
        for (const auto& [n, bit] : kVirtual) {
            if (key == n) {
                virtuals |= bit;
                return true;
            }
        }
        return false;
    }

    bool bind(const std::string& input, const std::string& targets) {
        Binding binding;
        binding.input = input;
        const auto name = lower(input);
        if (name.rfind("mouse", 0) == 0 && name.size() == 6 &&
            name[5] >= '1' && name[5] <= '5') {
            binding.source = Binding::Source::mouse_button;
            binding.code = name[5] - '0';
        } else if (name == "wheelup") {
            binding.source = Binding::Source::wheel_up;
        } else if (name == "wheeldown") {
            binding.source = Binding::Source::wheel_down;
        } else {
            const SDL_Scancode code = SDL_GetScancodeFromName(input.c_str());
            if (code == SDL_SCANCODE_UNKNOWN) {
                return false;
            }
            binding.source = Binding::Source::key;
            binding.code = code;
        }
        // "Enter = A, Plus" binds one input to several pad buttons.
        size_t start = 0;
        while (start <= targets.size()) {
            const auto comma = targets.find(',', start);
            const auto part = trim(targets.substr(
                start, comma == std::string::npos ? std::string::npos
                                                  : comma - start));
            if (!part.empty() &&
                !target(part, binding.buttons, binding.virtuals)) {
                return false;
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        bindings_.push_back(binding);
        return true;
    }

    void defaults() {
        static constexpr std::array<std::pair<const char*, const char*>, 26>
            kDefaults{{{"W", "MoveUp"},       {"S", "MoveDown"},
                       {"A", "MoveLeft"},     {"D", "MoveRight"},
                       {"Left Alt", "Walk"},  {"Mouse1", "ZR"},
                       {"Mouse2", "ZL"},      {"Mouse3", "StickR"},
                       {"Space", "B"},        {"Left Ctrl", "A"},
                       {"C", "A"},            {"R", "Y"},
                       {"F", "X"},            {"Q", "L"},
                       {"E", "R"},            {"Left Shift", "StickL"},
                       {"V", "StickR"},       {"Return", "A"},
                       {"Escape", "B"},       {"Backspace", "B"},
                       {"Tab", "Plus"},       {"P", "Plus"},
                       {"M", "Minus"},        {"Up", "Up"},
                       {"Down", "Down"},      {"Left", "Left"}}};
        for (const auto& [input, button] : kDefaults) {
            bind(input, button);
        }
        bind("Right", "Right");
        bind("WheelUp", "Up");
        bind("WheelDown", "Down");
        bind("1", "Left");
        bind("2", "Right");
    }

    void load() {
        const char* env = std::getenv("NWIIU_CONTROLS");
        const std::string path = env != nullptr ? env : "controls.ini";
        std::ifstream file(path);
        if (!file) {
            defaults();
            return;
        }
        std::string line;
        int number = 0;
        while (std::getline(file, line)) {
            ++number;
            if (const auto hash = line.find_first_of("#;");
                hash != std::string::npos) {
                line.resize(hash);
            }
            line = trim(line);
            if (line.empty() || line.front() == '[') {
                continue;
            }
            const auto equals = line.find('=');
            if (equals == std::string::npos) {
                std::fprintf(stderr, "controls: %s:%d: expected key = value\n",
                             path.c_str(), number);
                continue;
            }
            const auto key = trim(line.substr(0, equals));
            const auto value = trim(line.substr(equals + 1));
            const auto setting = lower(key);
            if (setting == "mouse_sensitivity") {
                sensitivity_ = std::atof(value.c_str());
            } else if (setting == "invert_y") {
                invert_y_ = lower(value) == "true" || value == "1";
            } else if (setting == "capture_mouse") {
                capture_mouse_ = !(lower(value) == "false" || value == "0");
            } else if (setting == "release_mouse_key") {
                release_key_ = SDL_GetScancodeFromName(value.c_str());
            } else if (setting == "gamepad_deadzone") {
                deadzone_ = std::clamp(
                    static_cast<float>(std::atof(value.c_str())), 0.f, 0.9f);
            } else if (!bind(key, value)) {
                std::fprintf(stderr, "controls: %s:%d: cannot bind '%s'\n",
                             path.c_str(), number, line.c_str());
            }
        }
        std::fprintf(stderr, "controls: loaded %zu bindings from %s\n",
                     bindings_.size(), path.c_str());
    }
};

} // namespace nwii::runtime
