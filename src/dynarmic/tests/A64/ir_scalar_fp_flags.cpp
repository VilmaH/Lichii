// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Scalar FP values that stay in vector registers across instructions, and NZCV
// values that are forwarded from flag setters to their readers inside a block.
// Every program is run with and without the IR optimisations, and the two
// architectural states must match each other and the expected values.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "common/common_types.h"

#include "dynarmic/tests/A64/testenv.h"
#include "dynarmic/tests/native/testenv.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/optimization_flags.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/opt_passes.h"

using namespace Dynarmic;

namespace {

struct Snapshot {
    std::array<u64, 31> regs;
    std::array<Vector, 32> vecs;
    u64 pc;
    u32 nzcv;
    u32 fpsr;
};

using Init = std::function<void(A64::Jit&)>;

// Without LLVM, Disassemble() reports the start and end of the emitted code.
std::optional<size_t> CodeEnd(const A64::Jit& jit) {
    const std::string text = jit.Disassemble();
    const auto pos = text.find("end: ");
    if (pos == std::string::npos)
        return std::nullopt;
    return std::stoull(text.substr(pos + 5), nullptr, 16);
}

Snapshot RunProgram(const std::vector<u32>& code, const Init& init, OptimizationFlag optimizations, size_t ticks, std::optional<size_t>* emitted = nullptr) {
    A64TestEnv env;
    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.optimizations = optimizations;
    A64::Jit jit{conf};

    env.code_mem = code;
    init(jit);
    jit.SetPC(0);
    env.ticks_left = ticks;

    const auto before = CodeEnd(jit);
    CheckedRun([&] { jit.Run(); });
    const auto after = CodeEnd(jit);
    if (emitted) {
        *emitted = before && after ? std::optional<size_t>{*after - *before} : std::nullopt;
    }

    return Snapshot{
        .regs = jit.GetRegisters(),
        .vecs = jit.GetVectors(),
        .pc = jit.GetPC(),
        .nzcv = jit.GetPstate() & 0xF0000000,
        .fpsr = jit.GetFpsr(),
    };
}

// The IR passes under test are off in the reference run. Block linking stays on: without it the
// dispatcher runs between blocks, which is not what is being compared here.
constexpr OptimizationFlag reference_optimizations = all_safe_optimizations & ~(OptimizationFlag::GetSetElimination | OptimizationFlag::ConstProp | OptimizationFlag::MiscIROpt);

Snapshot Run(const std::vector<u32>& code, const Init& init, size_t ticks = 100) {
    const Snapshot opt = RunProgram(code, init, all_safe_optimizations, ticks);
    const Snapshot ref = RunProgram(code, init, reference_optimizations, ticks);
    REQUIRE(opt.regs == ref.regs);
    for (size_t i = 0; i < opt.vecs.size(); ++i) {
        INFO("v" << i);
        REQUIRE(opt.vecs[i] == ref.vecs[i]);
    }
    REQUIRE(opt.pc == ref.pc);
    REQUIRE(opt.nzcv == ref.nzcv);
    REQUIRE(opt.fpsr == ref.fpsr);
    return opt;
}

// Signalling NaNs in the upper lanes catch any packed host operation that
// looks at lanes a scalar instruction must ignore (they would raise IOC).
constexpr Vector snan_upper{0x7F8000057F800006, 0x7FF0000000000007};

const std::vector<u32> scalar_chain{
    0x1e222820,  // fadd s0, s1, s2
    0x1e210803,  // fmul s3, s0, s1
    0x1f020064,  // fmadd s4, s3, s2, s0
    0x1e203885,  // fsub s5, s4, s0
    0x1e2040a6,  // fmov s6, s5
    0x1e22c0c7,  // fcvt d7, s6
    0x1e6728e8,  // fadd d8, d7, d7
    0x1e624109,  // fcvt s9, d8
    0x1e21c16a,  // fsqrt s10, s11
    0x1e380134,  // fcvtzs w20, s9
    0x1e220295,  // scvtf s21, w20
    0x1e6c2990,  // fadd d16, d12, d12
    0x1e302a11,  // fadd s17, s16, s16
    0x1e260212,  // fmov w18, s16
    0x1e270256,  // fmov s22, w18
    0x14000000,  // b .
};

void InitScalarChain(A64::Jit& jit) {
    for (size_t i = 0; i < 32; ++i)
        jit.SetVector(i, snan_upper);
    jit.SetVector(1, {0x7F8000013FC00000, 0xFFC000007F800002});  // s1 = 1.5
    jit.SetVector(2, {0x7F80000340000000, 0x7FF0000000000004});  // s2 = 2.0
    jit.SetVector(11, {0x7F80000940800000, 0x7F80000A7F80000B});  // s11 = 4.0
    jit.SetVector(12, {0x3FF0000040000000, 0x7FF0000000000001});  // d12, low word 0x40000000
    for (size_t i = 0; i < 31; ++i)
        jit.SetRegister(i, 0xDEAD0000DEAD0000 + i);
}

const std::vector<u32> scalar_vector_mix{
    0x1e222820,  // fadd s0, s1, s2
    0x4e21d403,  // fadd v3.4s, v0.4s, v1.4s
    0x1e230864,  // fmul s4, s3, s3
    0x6e0c0424,  // mov v4.s[1], v1.s[0]
    0x1e202885,  // fadd s5, s4, s0
    0x4e0404a6,  // dup v6.4s, v5.s[0]
    0x1e6b094c,  // fmul d12, d10, d11
    0x4e6ad58d,  // fadd v13.2d, v12.2d, v10.2d
    0x1e6b39ae,  // fsub d14, d13, d11
    0x1f4a35cf,  // fmadd d15, d14, d10, d13
    0x14000000,  // b .
};

void InitScalarVectorMix(A64::Jit& jit) {
    for (size_t i = 0; i < 32; ++i)
        jit.SetVector(i, snan_upper);
    jit.SetVector(1, {0x111111113FC00000, 0x3333333322222222});   // s1 = 1.5, normal upper lanes
    jit.SetVector(2, {0x7F80000340000000, 0x7FF0000000000004});   // s2 = 2.0
    jit.SetVector(10, {0x4004000000000000, 0xAAAAAAAAAAAAAAAA});  // d10 = 2.5
    jit.SetVector(11, {0x3FE0000000000000, 0x7FF0000000000001});  // d11 = 0.5
}

const std::vector<u32> fp_compare_select{
    0x1e222020,  // fcmp s1, s2
    0x1e224c23,  // fcsel s3, s1, s2, mi
    0x1e6bcd44,  // fcsel d4, d10, d11, gt
    0x1e232865,  // fadd s5, s3, s3
    0x14000000,  // b .
};

void InitFpCompareSelect(A64::Jit& jit) {
    InitScalarChain(jit);
    jit.SetVector(10, {0x4004000000000000, 0xAAAAAAAAAAAAAAAA});  // d10 = 2.5
    jit.SetVector(11, {0x3FE0000000000000, 0x7FF0000000000001});  // d11 = 0.5
}

const std::vector<u32> flags_set_read_set{
    0xeb01001f,  // cmp x0, x1
    0x9a84b062,  // csel x2, x3, x4, lt
    0x9a840465,  // csinc x5, x3, x4, eq
    0x9a04006c,  // adc x12, x3, x4
    0x6b000006,  // subs w6, w0, w0
    0x1a840067,  // csel w7, w3, w4, eq
    0xd53b420d,  // mrs x13, nzcv
    0xfa411002,  // ccmp x0, x1, #2, ne
    0x9a842468,  // csinc x8, x3, x4, cs
    0x9a04006e,  // adc x14, x3, x4
    0xab0a0149,  // adds x9, x10, x10
    0x9a84606b,  // csel x11, x3, x4, vs
    0x14000000,  // b .
};

const std::vector<u32> subs_bne_loop{
    0xf1000400,  // subs x0, x0, #1
    0x54ffffe1,  // b.ne 0
    0x14000000,  // b .
};

const std::vector<u32> cmp_blt{
    0xeb01001f,  // cmp x0, x1
    0x5400006b,  // b.lt 0x10
    0xd2800022,  // mov x2, #1
    0x14000000,  // b .
    0xd2800042,  // mov x2, #2
    0x14000000,  // b .
};

const std::vector<u32> fcmp_ccmp_bmi{
    0x1e6b2140,  // fcmp d10, d11
    0xfa41c008,  // ccmp x0, x1, #8, gt
    0x54000064,  // b.mi 0x14
    0xd2800022,  // mov x2, #1
    0x14000000,  // b .
    0xd2800042,  // mov x2, #2
    0x14000000,  // b .
};

const std::vector<u32> flags_across_blocks{
    0xeb01001f,  // cmp x0, x1
    0x14000001,  // b 0x8
    0x9a84b062,  // csel x2, x3, x4, lt
    0x9a83b465,  // cinc x5, x3, ge
    0x14000000,  // b .
};

const std::vector<u32> adds_reg_write_bcs{
    0x2b020020,  // adds w0, w1, w2
    0x91000463,  // add x3, x3, #1
    0x54000062,  // b.cs 0x14
    0xd2800022,  // mov x2, #1
    0x14000000,  // b .
    0xd2800042,  // mov x2, #2
    0x14000000,  // b .
};

Init Regs(std::vector<std::pair<size_t, u64>> values) {
    return [values](A64::Jit& jit) {
        for (const auto& [index, value] : values)
            jit.SetRegister(index, value);
    };
}

}  // namespace

TEST_CASE("A64 scalar FP: chain on s and d registers", "[a64][scalar_fp]") {
    const Snapshot s = Run(scalar_chain, InitScalarChain);

    REQUIRE(s.vecs[0] == Vector{0x40600000, 0});           // 3.5
    REQUIRE(s.vecs[3] == Vector{0x40A80000, 0});           // 5.25
    REQUIRE(s.vecs[4] == Vector{0x41600000, 0});           // 14.0
    REQUIRE(s.vecs[5] == Vector{0x41280000, 0});           // 10.5
    REQUIRE(s.vecs[6] == Vector{0x41280000, 0});           // 10.5
    REQUIRE(s.vecs[7] == Vector{0x4025000000000000, 0});   // 10.5
    REQUIRE(s.vecs[8] == Vector{0x4035000000000000, 0});   // 21.0
    REQUIRE(s.vecs[9] == Vector{0x41A80000, 0});           // 21.0
    REQUIRE(s.vecs[10] == Vector{0x40000000, 0});          // 2.0
    REQUIRE(s.regs[20] == 21);
    REQUIRE(s.vecs[21] == Vector{0x41A80000, 0});          // 21.0
    REQUIRE(s.vecs[16] == Vector{0x4000000040000000, 0});
    REQUIRE(s.vecs[17] == Vector{0x40800000, 0});          // 4.0
    REQUIRE(s.regs[18] == 0x40000000);
    REQUIRE(s.vecs[22] == Vector{0x40000000, 0});
    REQUIRE(s.vecs[1] == Vector{0x7F8000013FC00000, 0xFFC000007F800002});
    REQUIRE(s.vecs[11] == Vector{0x7F80000940800000, 0x7F80000A7F80000B});
    REQUIRE(s.vecs[31] == snan_upper);
    REQUIRE(s.fpsr == 0);
    REQUIRE(s.pc == 0x3c);
}

TEST_CASE("A64 scalar FP: mixed with vector ops on the same registers", "[a64][scalar_fp]") {
    const Snapshot s = Run(scalar_vector_mix, InitScalarVectorMix);

    REQUIRE(s.vecs[0] == Vector{0x40600000, 0});                                  // 3.5
    REQUIRE(s.vecs[3] == Vector{0x1111111140A00000, 0x3333333322222222});         // 5.0 | v1 upper
    REQUIRE(s.vecs[4] == Vector{0x3FC0000041C80000, 0});                          // 25.0, s[1] = 1.5
    REQUIRE(s.vecs[5] == Vector{0x41E40000, 0});                                  // 28.5
    REQUIRE(s.vecs[6] == Vector{0x41E4000041E40000, 0x41E4000041E40000});
    REQUIRE(s.vecs[12] == Vector{0x3FF4000000000000, 0});                         // 1.25
    REQUIRE(s.vecs[13] == Vector{0x400E000000000000, 0xAAAAAAAAAAAAAAAA});        // 3.75
    REQUIRE(s.vecs[14] == Vector{0x400A000000000000, 0});                         // 3.25
    REQUIRE(s.vecs[15] == Vector{0x4027C00000000000, 0});                         // 11.875
    REQUIRE(s.fpsr == 0);
}

TEST_CASE("A64 scalar FP: FCMP then FCSEL", "[a64][scalar_fp][nzcv]") {
    const Snapshot s = Run(fp_compare_select, InitFpCompareSelect);

    REQUIRE(s.nzcv == 0x80000000);
    REQUIRE(s.vecs[3] == Vector{0x3FC00000, 0});          // 1.5
    REQUIRE(s.vecs[4] == Vector{0x3FE0000000000000, 0});  // 0.5
    REQUIRE(s.vecs[5] == Vector{0x40400000, 0});          // 3.0
    REQUIRE(s.fpsr == 0);
}

TEST_CASE("A64 NZCV: set, read, set again within a block", "[a64][nzcv]") {
    const Snapshot s = Run(flags_set_read_set, Regs({{0, 5}, {1, 7}, {3, 0x33}, {4, 0x44}, {10, 0x8000000000000000}}));

    REQUIRE(s.regs[2] == 0x33);         // lt taken
    REQUIRE(s.regs[5] == 0x45);         // eq not taken
    REQUIRE(s.regs[12] == 0x77);        // C clear
    REQUIRE(s.regs[6] == 0);
    REQUIRE(s.regs[7] == 0x33);         // eq taken
    REQUIRE(s.regs[13] == 0x60000000);  // Z C
    REQUIRE(s.regs[8] == 0x33);         // ccmp failed, flags #2 → cs taken
    REQUIRE(s.regs[14] == 0x78);        // C set
    REQUIRE(s.regs[9] == 0);
    REQUIRE(s.regs[11] == 0x33);        // vs taken
    REQUIRE(s.nzcv == 0x70000000);      // Z C V
    REQUIRE(s.pc == 0x30);
}

TEST_CASE("A64 NZCV: SUBS + B.NE loop", "[a64][nzcv]") {
    const Snapshot s = Run(subs_bne_loop, Regs({{0, 3}}));

    REQUIRE(s.regs[0] == 0);
    REQUIRE(s.nzcv == 0x60000000);
    REQUIRE(s.pc == 8);
}

TEST_CASE("A64 NZCV: CMP + B.LT", "[a64][nzcv]") {
    SECTION("taken") {
        const Snapshot s = Run(cmp_blt, Regs({{0, 5}, {1, 7}}));
        REQUIRE(s.regs[2] == 2);
        REQUIRE(s.nzcv == 0x80000000);
        REQUIRE(s.pc == 0x14);
    }
    SECTION("not taken") {
        const Snapshot s = Run(cmp_blt, Regs({{0, 9}, {1, 7}}));
        REQUIRE(s.regs[2] == 1);
        REQUIRE(s.nzcv == 0x20000000);
        REQUIRE(s.pc == 0xc);
    }
}

TEST_CASE("A64 NZCV: FCMP + CCMP + B.MI", "[a64][nzcv][scalar_fp]") {
    const auto init = [](u64 d10, u64 x0) {
        return [=](A64::Jit& jit) {
            jit.SetVector(10, {d10, 0x7FF0000000000001});
            jit.SetVector(11, {0x3FE0000000000000, 0x7FF0000000000001});  // 0.5
            jit.SetRegister(0, x0);
            jit.SetRegister(1, 7);
        };
    };
    SECTION("gt, cmp sets N") {
        const Snapshot s = Run(fcmp_ccmp_bmi, init(0x4004000000000000, 5));
        REQUIRE(s.regs[2] == 2);
        REQUIRE(s.nzcv == 0x80000000);
        REQUIRE(s.pc == 0x18);
    }
    SECTION("gt, cmp clears N") {
        const Snapshot s = Run(fcmp_ccmp_bmi, init(0x4004000000000000, 9));
        REQUIRE(s.regs[2] == 1);
        REQUIRE(s.nzcv == 0x20000000);
        REQUIRE(s.pc == 0x10);
    }
    SECTION("not gt, immediate flags") {
        const Snapshot s = Run(fcmp_ccmp_bmi, init(0x3FD0000000000000, 9));
        REQUIRE(s.regs[2] == 2);
        REQUIRE(s.nzcv == 0x80000000);
        REQUIRE(s.pc == 0x18);
    }
}

TEST_CASE("A64 NZCV: flags live out of a block", "[a64][nzcv]") {
    const Snapshot s = Run(flags_across_blocks, Regs({{0, 5}, {1, 7}, {3, 0x33}, {4, 0x44}}));

    REQUIRE(s.regs[2] == 0x33);
    REQUIRE(s.regs[5] == 0x33);
    REQUIRE(s.nzcv == 0x80000000);
    REQUIRE(s.pc == 0x10);
}

TEST_CASE("A64 NZCV: ADDS, register write, B.CS", "[a64][nzcv]") {
    SECTION("carry") {
        const Snapshot s = Run(adds_reg_write_bcs, Regs({{0, ~u64(0)}, {1, 0xFFFFFFFF}, {2, 1}, {3, 0x33}}));
        REQUIRE(s.regs[0] == 0);
        REQUIRE(s.regs[2] == 2);
        REQUIRE(s.regs[3] == 0x34);
        REQUIRE(s.nzcv == 0x60000000);
        REQUIRE(s.pc == 0x18);
    }
    SECTION("no carry") {
        const Snapshot s = Run(adds_reg_write_bcs, Regs({{1, 1}, {2, 1}, {3, 0x33}}));
        REQUIRE(s.regs[0] == 2);
        REQUIRE(s.regs[2] == 1);
        REQUIRE(s.nzcv == 0);
        REQUIRE(s.pc == 0x10);
    }
}

// Run with "[.codesize]" to print the host code emitted for each program.
TEST_CASE("A64 scalar FP / NZCV: emitted code size", "[.codesize]") {
    const auto report = [](const char* name, const std::vector<u32>& code, const Init& init) {
        std::optional<size_t> emitted;
        RunProgram(code, init, all_safe_optimizations, 100, &emitted);
        if (emitted)
            std::printf("%-22s %5zu bytes\n", name, *emitted);
    };
    report("scalar_chain", scalar_chain, InitScalarChain);
    report("scalar_vector_mix", scalar_vector_mix, InitScalarVectorMix);
    report("fp_compare_select", fp_compare_select, InitFpCompareSelect);
    report("flags_set_read_set", flags_set_read_set, Regs({{0, 5}, {1, 7}}));
    report("subs_bne_loop", subs_bne_loop, Regs({{0, 3}}));
    report("cmp_blt", cmp_blt, Regs({{0, 5}, {1, 7}}));
    report("fcmp_ccmp_bmi", fcmp_ccmp_bmi, Regs({{0, 5}, {1, 7}}));
    report("adds_reg_write_bcs", adds_reg_write_bcs, Regs({{1, 1}, {2, 1}}));
}

namespace {

const A64::LocationDescriptor block_location{0, FP::FPCR{}};

// Fills `block` with the block at PC 0 after the optimisation passes, as the x64 backend would receive it.
void Optimize(IR::Block& block, const std::vector<u32>& code) {
    A64::Translate(block, block_location, [&](u64 vaddr) -> std::optional<u32> { return code.at(vaddr / 4); }, {});
    A64::UserConfig conf{};
    Optimization::Optimize(block, conf, {.forward_nzcv = true});
}

size_t Count(const IR::Block& block, std::initializer_list<IR::Opcode> opcodes) {
    size_t n = 0;
    for (const auto& inst : block.instructions)
        n += std::find(opcodes.begin(), opcodes.end(), inst.GetOpcode()) != opcodes.end();
    return n;
}

}  // namespace

TEST_CASE("A64 IR: scalar register round trips are folded", "[a64][scalar_fp][ir]") {
    IR::Block block{block_location};
    Optimize(block, scalar_chain);
    for (const auto& inst : block.instructions) {
        if (inst.GetOpcode() != IR::Opcode::VectorGetElement32 && inst.GetOpcode() != IR::Opcode::VectorGetElement64)
            continue;
        // Only reads of registers that are live into the block remain.
        INFO(IR::DumpBlock(block));
        REQUIRE(inst.GetArg(0).GetInst()->GetOpcode() == IR::Opcode::A64GetQ);
    }
    REQUIRE(Count(block, {IR::Opcode::ZeroExtendLongToQuad}) <= Count(block, {IR::Opcode::A64SetQ}));
}

TEST_CASE("A64 IR: in-block NZCV is passed to its readers", "[a64][nzcv][ir]") {
    IR::Block block{block_location};
    Optimize(block, flags_set_read_set);
    INFO(IR::DumpBlock(block));
    REQUIRE(Count(block, {IR::Opcode::ConditionalSelect32, IR::Opcode::ConditionalSelect64, IR::Opcode::ConditionalSelectNZCV}) == 0);
    REQUIRE(Count(block, {IR::Opcode::ConditionalSelectWithFlags32, IR::Opcode::ConditionalSelectWithFlags64, IR::Opcode::ConditionalSelectWithFlagsNZCV}) == 6);
    REQUIRE(Count(block, {IR::Opcode::A64GetCFlag}) == 0);
    // mrs nzcv still reads the state, so the store before it stays, as does the final one.
    REQUIRE(Count(block, {IR::Opcode::A64SetNZCV}) == 2);
}

TEST_CASE("A64 IR: the final NZCV store ends a conditional block", "[a64][nzcv][ir]") {
    IR::Block block{block_location};
    Optimize(block, adds_reg_write_bcs);
    INFO(IR::DumpBlock(block));
    REQUIRE(block.instructions.back().GetOpcode() == IR::Opcode::A64SetNZCV);
    REQUIRE(Count(block, {IR::Opcode::A64SetNZCV}) == 1);
}
