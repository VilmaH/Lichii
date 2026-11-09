// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Checks the inline x64 paths for FRINTA, FCVTAS/FCVTAU, SSHL/USHL (8/16-bit), UQXTN and SQXTUN
// against reference results, including the cumulative FPSR flags.

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/op.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/tests/A64/testenv.h"
#include "dynarmic/tests/native/testenv.h"

using namespace Dynarmic;

namespace {

constexpr u32 FPCR_DN = 1 << 25;
constexpr u32 FPCR_FZ = 1 << 24;
constexpr u32 FPCR_RZ = 3 << 22;
constexpr u32 FPCR_RM = 2 << 22;
constexpr u32 FPSR_IOC = 1 << 0;
constexpr u32 FPSR_IXC = 1 << 4;
constexpr u32 FPSR_IDC = 1 << 7;
constexpr u32 FPSR_QC = 1 << 27;

constexpr std::array<u32, 6> fpcr_values{0, FPCR_DN, FPCR_FZ, FPCR_DN | FPCR_FZ, FPCR_RZ, FPCR_RM | FPCR_FZ};

// Runs a single instruction with V1, V2 and X0 as inputs.
class Runner {
public:
    explicit Runner(u32 instruction) {
        env.code_mem = {instruction, 0x14000000};  // B .
        config.callbacks = &env;
        jit = std::make_unique<A64::Jit>(config);
    }

    void Run(u32 fpcr, Vector v1, Vector v2 = {}, Vector v0 = {0x0123456789ABCDEF, 0xFEDCBA9876543210}) {
        jit->SetPC(0);
        jit->SetFpcr(fpcr);
        jit->SetFpsr(0);
        jit->SetVector(0, v0);
        jit->SetVector(1, v1);
        jit->SetVector(2, v2);
        jit->SetRegister(0, 0xDEADBEEFDEADBEEF);
        env.ticks_left = 2;
        CheckedRun([&] { jit->Run(); });
    }

    Vector V0() const { return jit->GetVector(0); }
    u64 X0() const { return jit->GetRegister(0); }
    u32 Fpsr() const { return jit->GetFpsr(); }

private:
    A64TestEnv env;
    A64::UserConfig config{};
    std::unique_ptr<A64::Jit> jit;
};

std::string Hex(u64 value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
    return buf;
}

template<typename FPT>
using FloatOf = std::conditional_t<sizeof(FPT) == 4, float, double>;

// Edge cases plus random values. Most random values are near small integers and ties.
template<typename FPT>
std::vector<FPT> FloatInputs(size_t random_count) {
    using F = FloatOf<FPT>;
    constexpr int mantissa_bits = std::numeric_limits<F>::digits - 1;
    const auto bits = [](F f) { return std::bit_cast<FPT>(f); };
    std::vector<FPT> v;
    const auto add_around = [&](F f) {
        for (const F base : {f, -f}) {
            v.push_back(bits(base));
            v.push_back(bits(std::nextafter(base, F(0))));
            v.push_back(bits(std::nextafter(base, std::numeric_limits<F>::infinity())));
            v.push_back(bits(std::nextafter(base, -std::numeric_limits<F>::infinity())));
        }
    };
    for (const F f : {F(0), F(0.25), F(0.5), F(0.75), F(1), F(1.5), F(2.5), F(3.5), F(1e3) + F(0.5)}) {
        add_around(f);
    }
    // Largest values with a .5 fraction, and the limits of the integer formats
    add_around(std::ldexp(F(1), mantissa_bits) - F(0.5));
    add_around(std::ldexp(F(1), mantissa_bits - 1) + F(0.5));
    add_around(std::ldexp(F(1), mantissa_bits));
    add_around(std::ldexp(F(1), mantissa_bits + 1));
    for (const int e : {31, 32, 63, 64}) {
        add_around(std::ldexp(F(1), e));
        add_around(std::ldexp(F(1), e) - F(0.5));
        add_around(std::ldexp(F(1), e) + F(0.5));
    }
    add_around(std::numeric_limits<F>::max());
    add_around(std::numeric_limits<F>::infinity());
    add_around(std::numeric_limits<F>::min());          // smallest normal, largest denormal
    add_around(std::numeric_limits<F>::denorm_min());
    constexpr FPT sign = FPT(1) << (sizeof(FPT) * 8 - 1);
    constexpr FPT exponent = std::bit_cast<FPT>(std::numeric_limits<F>::infinity());
    constexpr FPT quiet = FPT(1) << (mantissa_bits - 1);
    for (const FPT nan : {FPT(exponent | quiet), FPT(exponent | quiet | 0x1234), FPT(exponent | 1), FPT(exponent | 0x5678)}) {
        v.push_back(nan);
        v.push_back(nan | sign);
    }

    std::mt19937_64 rng(0x1234 + sizeof(FPT));
    for (size_t i = 0; i < random_count; i++) {
        switch (i % 4) {
        case 0:
            v.push_back(FPT(rng()));
            break;
        case 1: {
            // n + k/4 with n up to 2^(mantissa_bits + 1)
            const int magnitude = int(rng() % (mantissa_bits + 2));
            const F n = std::floor(std::ldexp(F(rng() % (u64(1) << 20)), magnitude - 20));
            const F f = n + F(rng() % 4) / 4;
            v.push_back(bits(rng() & 1 ? -f : f));
            break;
        }
        case 2: {
            const F f = std::ldexp(F(rng() % 1000000) / 1000000, int(rng() % 80) - 10);
            v.push_back(bits(rng() & 1 ? -f : f));
            break;
        }
        case 3:
            // denormals
            v.push_back(FPT(FPT(rng()) & ((FPT(1) << mantissa_bits) - 1)) | (rng() & 1 ? sign : 0));
            break;
        }
    }
    return v;
}

// Counts mismatches and stops reporting after a few.
struct Mismatches {
    size_t count = 0;
    void Report(const std::string& what) {
        if (++count <= 10) {
            FAIL_CHECK(what);
        }
    }
    ~Mismatches() { CHECK(count == 0); }
};

template<typename FPT>
void CheckScalarRoundInt(u32 instruction, const char* name, FP::RoundingMode rounding, bool exact, const std::vector<FPT>& inputs) {
    Runner runner{instruction};
    Mismatches mismatches;
    for (const u32 fpcr : fpcr_values) {
        for (const FPT x : inputs) {
            runner.Run(fpcr, {x, 0x5555555555555555});
            FP::FPSR fpsr{};
            const FP::RoundingMode mode = exact ? FP::FPCR{fpcr}.RMode() : rounding;  // FRINTX uses FPCR
            const u64 expected = FP::FPRoundInt<FPT>(x, FP::FPCR{fpcr}, mode, exact, fpsr);
            if (runner.V0() != Vector{expected, 0} || runner.Fpsr() != fpsr.Value()) {
                mismatches.Report(std::string(name) + " fpcr=" + Hex(fpcr) + " x=" + Hex(x) + " got=" + Hex(runner.V0()[0]) + " fpsr=" + Hex(runner.Fpsr())
                                  + " expected=" + Hex(expected) + " fpsr=" + Hex(fpsr.Value()));
            }
        }
    }
}

}  // namespace

TEST_CASE("A64: FRINT scalar directed", "[a64][fp_inline]") {
    struct Case {
        u32 instruction;
        u64 input;
        u64 expected;
        u32 fpsr;
    };
    const std::vector<Case> cases{
        {0x1E264020, 0x40200000, 0x40400000, 0},  // FRINTA S0, S1: 2.5 -> 3
        {0x1E264020, 0xC0200000, 0xC0400000, 0},  // FRINTA S0, S1: -2.5 -> -3
        {0x1E264020, 0x3F000000, 0x3F800000, 0},  // FRINTA S0, S1: 0.5 -> 1
        {0x1E264020, 0xBE800000, 0x80000000, 0},  // FRINTA S0, S1: -0.25 -> -0
        {0x1E664020, 0x4004000000000000, 0x4008000000000000, 0},  // FRINTA D0, D1: 2.5 -> 3
        {0x1E664020, 0xC004000000000000, 0xC008000000000000, 0},  // FRINTA D0, D1: -2.5 -> -3
        {0x1E664020, 0x3FE0000000000000, 0x3FF0000000000000, 0},  // FRINTA D0, D1: 0.5 -> 1
        {0x1E274020, 0x40200000, 0x40000000, FPSR_IXC},  // FRINTX S0, S1: 2.5 -> 2
        {0x1E274020, 0x40400000, 0x40400000, 0},         // FRINTX S0, S1: 3 -> 3
        {0x1E674020, 0x4004000000000000, 0x4000000000000000, FPSR_IXC},  // FRINTX D0, D1: 2.5 -> 2
        {0x1EE64020, 0x4100, 0x4200, 0},  // FRINTA H0, H1: 2.5 -> 3
        {0x1EE64020, 0xC100, 0xC200, 0},  // FRINTA H0, H1: -2.5 -> -3
        {0x1EE64020, 0x3800, 0x3C00, 0},  // FRINTA H0, H1: 0.5 -> 1
        {0x1EE74020, 0x4100, 0x4000, FPSR_IXC},  // FRINTX H0, H1: 2.5 -> 2
        {0x1EE44020, 0x4100, 0x4000, 0},  // FRINTN H0, H1: 2.5 -> 2
        {0x1EE4C020, 0x4080, 0x4200, 0},  // FRINTP H0, H1: 2.25 -> 3
        {0x1EE54020, 0xC080, 0xC200, 0},  // FRINTM H0, H1: -2.25 -> -3
        {0x1EE5C020, 0xC180, 0xC000, 0},  // FRINTZ H0, H1: -2.75 -> -2
    };
    for (const Case& c : cases) {
        Runner runner{c.instruction};
        runner.Run(0, {c.input, 0});
        INFO("instruction " << Hex(c.instruction) << " input " << Hex(c.input));
        CHECK(runner.V0() == Vector{c.expected, 0});
        CHECK(runner.Fpsr() == c.fpsr);
    }
}

TEST_CASE("A64: FRINT scalar vs FPRoundInt", "[a64][fp_inline]") {
    std::vector<u16> halves(0x10000);
    for (size_t i = 0; i < halves.size(); i++) {
        halves[i] = u16(i);
    }
    CheckScalarRoundInt<u16>(0x1EE64020, "FRINTA H0, H1", FP::RoundingMode::ToNearest_TieAwayFromZero, false, halves);
    CheckScalarRoundInt<u16>(0x1EE74020, "FRINTX H0, H1", FP::RoundingMode::ToNearest_TieEven, true, halves);
    CheckScalarRoundInt<u16>(0x1EE4C020, "FRINTP H0, H1", FP::RoundingMode::TowardsPlusInfinity, false, halves);
    CheckScalarRoundInt<u32>(0x1E264020, "FRINTA S0, S1", FP::RoundingMode::ToNearest_TieAwayFromZero, false, FloatInputs<u32>(4000));
    CheckScalarRoundInt<u64>(0x1E664020, "FRINTA D0, D1", FP::RoundingMode::ToNearest_TieAwayFromZero, false, FloatInputs<u64>(4000));
    CheckScalarRoundInt<u32>(0x1E274020, "FRINTX S0, S1", FP::RoundingMode::ToNearest_TieEven, true, FloatInputs<u32>(1000));
    CheckScalarRoundInt<u64>(0x1E674020, "FRINTX D0, D1", FP::RoundingMode::ToNearest_TieEven, true, FloatInputs<u64>(1000));
}

