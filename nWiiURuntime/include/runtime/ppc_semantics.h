#pragma once

#include "runtime/cpu_context.h"
#include "runtime/memory.h"

#include <cstdint>

namespace nwii::runtime::ppc {
constexpr bool valid_lswi_form(uint32_t rt, uint32_t ra,
                               uint32_t byte_count) {
    const uint32_t register_count =
        ((byte_count == 0 ? 32 : byte_count) + 3) / 4;
    return !(rt == 0 && ra == 0) &&
           (ra == 0 || ((ra - rt) & 31) >= register_count);
}

// Accounting only: what recompiled blocks need on the hot path.
inline void count_instruction(CPUContext& cpu) { ++cpu.instruction_count; }

inline void trace_instruction(CPUContext& cpu, uint32_t raw_instruction) {
    cpu.current_instruction = raw_instruction;
    cpu.pc_history[cpu.history_cursor] = cpu.pc;
    cpu.history_cursor = (cpu.history_cursor + 1) % cpu.pc_history.size();
    if (cpu.history_size < cpu.pc_history.size()) {
        ++cpu.history_size;
    }
    ++cpu.instruction_count;
}

void mulli(CPUContext& cpu, uint32_t rt, uint32_t ra, int32_t immediate);
void subfic(CPUContext& cpu, uint32_t rt, uint32_t ra, int32_t immediate);
void cmpli(CPUContext& cpu, uint32_t bf, uint32_t ra, uint32_t immediate);
void cmpi(CPUContext& cpu, uint32_t bf, uint32_t ra, int32_t immediate);
void addic(CPUContext& cpu, uint32_t rt, uint32_t ra, int32_t immediate,
           bool record);
void addi(CPUContext& cpu, uint32_t rt, uint32_t ra, int32_t immediate);
void addis(CPUContext& cpu, uint32_t rt, uint32_t ra, int32_t immediate);
void rlwinm(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t sh,
            uint32_t mb, uint32_t me, bool rc);
void rlwnm(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb,
           uint32_t mb, uint32_t me, bool rc);
void rlwimi(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t sh,
            uint32_t mb, uint32_t me, bool rc);
void ori(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t immediate);
void xori(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t immediate);
void andi_dot(CPUContext& cpu, uint32_t ra, uint32_t rs,
              uint32_t immediate);

void cmp(CPUContext& cpu, uint32_t bf, uint32_t ra, uint32_t rb);
void cmpl(CPUContext& cpu, uint32_t bf, uint32_t ra, uint32_t rb);
void tw(CPUContext& cpu, uint32_t to, uint32_t ra, uint32_t rb);
void mfcr(CPUContext& cpu, uint32_t rt);
void subf(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void subfc(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void subfe(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void subfze(CPUContext& cpu, uint32_t rt, uint32_t ra, bool oe, bool rc);
void neg(CPUContext& cpu, uint32_t rt, uint32_t ra, bool rc);
void cntlzw(CPUContext& cpu, uint32_t ra, uint32_t rs, bool rc);
void and_(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void andc(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void nor_(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void mulhwu(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void mulhw(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void mullw(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void divwu(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void divw(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool oe,
          bool rc);
void add(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void addc(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool rc);
void addze(CPUContext& cpu, uint32_t rt, uint32_t ra, bool oe, bool rc);
void adde(CPUContext& cpu, uint32_t rt, uint32_t ra, uint32_t rb, bool oe,
          bool rc);
void or_(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void orc(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void xor_(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void srawi(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t sh, bool rc);
void sraw(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void slw(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void srw(CPUContext& cpu, uint32_t ra, uint32_t rs, uint32_t rb, bool rc);
void extsb(CPUContext& cpu, uint32_t ra, uint32_t rs, bool rc);
void extsh(CPUContext& cpu, uint32_t ra, uint32_t rs, bool rc);
void mfspr(CPUContext& cpu, uint32_t rt, uint32_t spr);
void mtspr(CPUContext& cpu, uint32_t rs, uint32_t spr);

template <class Memory>
void lwz(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void lwzu(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lhzu(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lhz(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void lha(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void lhau(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lbz(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void lbzu(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          int32_t displacement);
template <class Memory>
void stw(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
         int32_t displacement);
template <class Memory>
void stwu(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void stb(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
         int32_t displacement);
template <class Memory>
void stbu(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void sth(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
         int32_t displacement);
template <class Memory>
void sthu(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lmw(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void stmw(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lwzx(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lwzux(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lwarx(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lwbrx(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lbzux(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lhaux(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lbzx(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lhax(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lhzx(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lhzux(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void stwx(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          uint32_t rb);
template <class Memory>
void stwux(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
           uint32_t rb);
template <class Memory>
void stwcx(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
           uint32_t rb);
template <class Memory>
void stwbrx(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
            uint32_t rb);
template <class Memory>
void stbux(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
           uint32_t rb);
template <class Memory>
void sthux(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
           uint32_t rb);
void dcbf(CPUContext& cpu);
void dcbst(CPUContext& cpu);
template <class Memory>
void dcbz(CPUContext& cpu, Memory& memory, uint32_t ra, uint32_t rb);
template <class Memory>
void stbx(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          uint32_t rb);
template <class Memory>
void sthx(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lswi(CPUContext& cpu, Memory& memory, uint32_t rt, uint32_t ra,
          uint32_t byte_count);
template <class Memory>
void stswi(CPUContext& cpu, Memory& memory, uint32_t rs, uint32_t ra,
           uint32_t byte_count);
template <class Memory>
void stfiwx(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
            uint32_t rb);
template <class Memory>
void stfsx(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
           uint32_t rb);
template <class Memory>
void lfsx(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
          uint32_t rb);
template <class Memory>
void lfsux(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
           uint32_t rb);
template <class Memory>
void stfsux(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
            uint32_t rb);
template <class Memory>
void psq_l(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
           bool w, uint32_t gqr, int32_t displacement);
template <class Memory>
void psq_st(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
            bool w, uint32_t gqr, int32_t displacement);
template <class Memory>
void psq_lu(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
            bool w, uint32_t gqr, int32_t displacement);
template <class Memory>
void psq_stu(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
             bool w, uint32_t gqr, int32_t displacement);

template <class Memory>
void lfs(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void lfsu(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
          int32_t displacement);
template <class Memory>
void lfd(CPUContext& cpu, Memory& memory, uint32_t frt, uint32_t ra,
         int32_t displacement);
template <class Memory>
void stfs(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void stfsu(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
           int32_t displacement);
void fdivs(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void fres(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fadds(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void fmuls(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc);
void fmul(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc);
void fmadds(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
            uint32_t frc);
void fmadd(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
           uint32_t frc);
void fnmsub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
           uint32_t frc);
void fmsubs(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
            uint32_t frc);
void fnmsubs(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
             uint32_t frc);
void fmsub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb,
           uint32_t frc);
template <class Memory>
void stfd(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
          int32_t displacement);
template <class Memory>
void stfdu(CPUContext& cpu, Memory& memory, uint32_t frs, uint32_t ra,
           int32_t displacement);
void fsubs(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void fsub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void fadd(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void fdiv(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void frsp(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fabs(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fneg(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fnabs(CPUContext& cpu, uint32_t frt, uint32_t frb);
void frsqrte(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fctiw(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fctiwz(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fmr(CPUContext& cpu, uint32_t frt, uint32_t frb);
void fcmpu(CPUContext& cpu, uint32_t bf, uint32_t fra, uint32_t frb);
void fsel(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
          uint32_t frb);
void ps_merge00(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_merge10(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_merge01(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_merge11(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_neg(CPUContext& cpu, uint32_t frt, uint32_t frb);
void ps_muls0(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc);
void ps_mul(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc);
void ps_muls1(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc);
void ps_madds0(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
               uint32_t frb);
void ps_madds1(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
               uint32_t frb);
void ps_madd(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
             uint32_t frb);
void ps_sum0(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
             uint32_t frb);
void ps_sum1(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
             uint32_t frb);
void ps_add(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_sub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frb);
void ps_msub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
             uint32_t frb);
void ps_sel(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
            uint32_t frb);
void ps_cmpo0(CPUContext& cpu, uint32_t bf, uint32_t fra, uint32_t frb);
void ps_rsqrte(CPUContext& cpu, uint32_t frt, uint32_t frb);
void ps_nmsub(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
              uint32_t frb);
void ps_nmadd(CPUContext& cpu, uint32_t frt, uint32_t fra, uint32_t frc,
              uint32_t frb);

template <class Memory>
uint32_t relocated_branch_target(const Memory& memory, uint32_t address,
                                 uint32_t original, uint32_t fallback);
void b(CPUContext& cpu, uint32_t target, bool link);
void bc(CPUContext& cpu, uint32_t target, uint32_t bo, uint32_t bi,
        bool link);
void bclr(CPUContext& cpu, uint32_t bo, uint32_t bi, bool link);
void bcctr(CPUContext& cpu, uint32_t bo, uint32_t bi, bool link);
void isync(CPUContext& cpu);
void crxor(CPUContext& cpu, uint32_t bt, uint32_t ba, uint32_t bb);
void creqv(CPUContext& cpu, uint32_t bt, uint32_t ba, uint32_t bb);
} // namespace nwii::runtime::ppc

// Definitions are header-inline on purpose. Recompiled blocks call one of
// these per guest instruction; out of line in their own translation unit each
// became a PLT call, which blocked inlining and forced every guest register
// through memory across it. A five instruction block emitted eighteen calls.
#include "runtime/ppc_semantics_inl.h"
