#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 || i1*Q + Q > args.ne31 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}
// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
kernel void kernel_flash_attn_ext_vec_idx(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

// Sparse flash attention with the heads as the matrix rows: one threadgroup per (query row, KV head).
// The G query heads that share a KV head form the rows of the Q tile, so each gathered K/V row is
// loaded once for all G heads and the products run on simdgroup matrices. Walks the index lists
// built by kernel_flash_attn_ext_vec_idx (ascending, -1 padded). F16 K/V, DK == DV == D, G <= 16,
// mask shared by all heads.
template<short D, short NSG>
kernel void kernel_flash_attn_ext_sparse_hr(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const int  * idx,
        device       char * dst,
        threadgroup  half * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NT  = NSG*N_SIMDWIDTH;
    constexpr short R   = 16;     // Q rows (heads), two 8-row blocks
    constexpr short RS  = R/NSG;  // softmax rows per simdgroup
    constexpr short C   = 32;     // gathered cells per step
    constexpr short DS  = D/NSG;  // O columns per simdgroup
    constexpr short NO  = DS/8;

    const int iq1 = tgpig[0];
    const int ikv = tgpig[1];
    const int iq3 = tgpig[2];

    const int G    = args.ne02/args.ne_12_2;
    const int ikv3 = iq3/(args.ne03/args.ne_12_3);

    threadgroup half  * sq    = shmem;                                 // [R][D]
    threadgroup half  * sk    = sq + R*D;                              // [C][D] K, then V
    threadgroup float * ss    = (threadgroup float *) (sk + C*D);      // [R][C] scores, then P
    threadgroup float * sdiag = ss + R*C;                              // [2][8][8] rescale diagonals
    threadgroup float * sms   = sdiag + 2*64;                          // [R] rescale factors, then sums
    threadgroup int   * sid   = (threadgroup int *) (sms + R);         // [C] cell ids
    threadgroup half  * smk   = (threadgroup half *) (sid + C);        // [C] mask values

    device const int  * pidx = idx + (((int64_t) (iq3%args.ne33)*args.ne32)*args.ne31 + iq1)*args.n_kv_max_padded;
    device const half * pm   = (device const half *) (mask + iq1*args.nb31 + (iq3%args.ne33)*args.nb33);

    device const char * kb = k + ikv*args.nb12 + ikv3*args.nb13;
    device const char * vb = v + ikv*args.nb22 + ikv3*args.nb23;

    for (int i = tiitg; i < R*D/4; i += NT) {
        const int r  = i/(D/4);
        const int c4 = i%(D/4);

        half4 val = 0;
        if (r < G) {
            device const float4 * q4 = (device const float4 *) (q + iq1*args.nb01 + (ikv*G + r)*args.nb02 + iq3*args.nb03);
            val = (half4) q4[c4];
        }
        ((threadgroup half4 *) sq)[i] = val;
    }

    for (int i = tiitg; i < 2*64; i += NT) {
        sdiag[i] = 0.0f;
    }

    float Mr[RS];
    float Sr[RS];
    for (short jj = 0; jj < RS; ++jj) {
        Mr[jj] = -FLT_MAX/2;
        Sr[jj] = 0.0f;
    }

    simdgroup_float8x8 lo[2][NO];
    for (short ii = 0; ii < NO; ++ii) {
        lo[0][ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        lo[1][ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int j0 = 0; j0 < args.n_kv_max_padded; j0 += C) {
        if (tiitg < C) {
            const int id = pidx[j0 + tiitg];
            sid[tiitg] = id;
            smk[tiitg] = id >= 0 ? pm[id] : (half) -MAXHALF;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the lists are ascending with the -1 padding at the tail
        if (sid[0] < 0) {
            break;
        }

        for (int i = tiitg; i < C*D/8; i += NT) {
            const int c  = i/(D/8);
            const int d8 = i%(D/8);
            const int id = sid[c];

            uint4 val = 0;
            if (id >= 0) {
                val = ((device const uint4 *) (kb + (uint64_t) id*args.nb11))[d8];
            }
            ((threadgroup uint4 *) sk)[i] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q*K^T: with 4 simdgroups, simdgroup sgitg owns the cells [8*sgitg, 8*sgitg + 8) for
        // both row blocks; with 8, it owns one (row block, cell tile) pair
        if (NSG == 4) {
            simdgroup_float8x8 mqk[2] = {
                make_filled_simdgroup_matrix<float, 8>(0.0f),
                make_filled_simdgroup_matrix<float, 8>(0.0f),
            };

            #pragma unroll (4)
            for (short i = 0; i < D/8; ++i) {
                simdgroup_half8x8 mk;
                simdgroup_half8x8 mq0;
                simdgroup_half8x8 mq1;

                simdgroup_load(mk,  sk + (8*sgitg)*D + 8*i, D, 0, true);
                simdgroup_load(mq0, sq + 0*8*D + 8*i, D);
                simdgroup_load(mq1, sq + 1*8*D + 8*i, D);

                simdgroup_multiply_accumulate(mqk[0], mq0, mk, mqk[0]);
                simdgroup_multiply_accumulate(mqk[1], mq1, mk, mqk[1]);
            }

            simdgroup_store(mqk[0], ss + 0*8*C + 8*sgitg, C, 0, false);
            simdgroup_store(mqk[1], ss + 1*8*C + 8*sgitg, C, 0, false);
        } else {
            const short rb = sgitg/4;
            const short ct = sgitg%4;

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            #pragma unroll (4)
            for (short i = 0; i < D/8; ++i) {
                simdgroup_half8x8 mk;
                simdgroup_half8x8 mq;

                simdgroup_load(mk, sk + (8*ct)*D + 8*i, D, 0, true);
                simdgroup_load(mq, sq + rb*8*D + 8*i, D);

                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }

            simdgroup_store(mqk, ss + rb*8*C + 8*ct, C, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // stage V over K while the online softmax runs
        for (int i = tiitg; i < C*D/8; i += NT) {
            const int c  = i/(D/8);
            const int d8 = i%(D/8);
            const int id = sid[c];

            uint4 val = 0;
            if (id >= 0) {
                val = ((device const uint4 *) (vb + (uint64_t) id*args.nb21))[d8];
            }
            ((threadgroup uint4 *) sk)[i] = val;
        }

        for (short jj = 0; jj < RS; ++jj) {
            const short j = sgitg*RS + jj;

            const float s = ss[j*C + tiisg]*args.scale + (float) smk[tiisg];
            const float m = Mr[jj];

            Mr[jj] = simd_max(max(m, s));

            const float ms = exp(m - Mr[jj]);
            const float p  = exp(s - Mr[jj]);

            Sr[jj] = Sr[jj]*ms + simd_sum(p);

            ss[j*C + tiisg] = p;

            if (tiisg == 0) {
                sms[j] = ms;
                sdiag[(j/8)*64 + (j%8)*9] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool rescale = false;
        for (short j = 0; j < R; ++j) {
            rescale |= sms[j] != 1.0f;
        }

        if (rescale) {
            simdgroup_float8x8 md0;
            simdgroup_float8x8 md1;

            simdgroup_load(md0, sdiag + 0,  8);
            simdgroup_load(md1, sdiag + 64, 8);

            for (short ii = 0; ii < NO; ++ii) {
                simdgroup_multiply(lo[0][ii], md0, lo[0][ii]);
                simdgroup_multiply(lo[1][ii], md1, lo[1][ii]);
            }
        }

        // O = O + P*V: simdgroup sgitg owns the columns [DS*sgitg, DS*sgitg + DS)
        FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
            simdgroup_float8x8 vs0;
            simdgroup_float8x8 vs1;

            simdgroup_load(vs0, ss + 0*8*C + 8*cc, C);
            simdgroup_load(vs1, ss + 1*8*C + 8*cc, C);

            FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                simdgroup_half8x8 mv;

                simdgroup_load(mv, sk + (8*cc)*D + DS*sgitg + 8*ii, D);

                simdgroup_multiply_accumulate(lo[0][ii], vs0, mv, lo[0][ii]);
                simdgroup_multiply_accumulate(lo[1][ii], vs1, mv, lo[1][ii]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // the K/V staging area holds R*D floats exactly
    threadgroup float * so = (threadgroup float *) sk;

    for (short ii = 0; ii < NO; ++ii) {
        simdgroup_store(lo[0][ii], so + 0*8*D + DS*sgitg + 8*ii, D);
        simdgroup_store(lo[1][ii], so + 1*8*D + DS*sgitg + 8*ii, D);
    }

    if (tiisg == 0) {
        for (short jj = 0; jj < RS; ++jj) {
            sms[sgitg*RS + jj] = Sr[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tiitg; i < G*D/4; i += NT) {
        const int r  = i/(D/4);
        const int c4 = i%(D/4);

        const float s     = sms[r];
        const float scale = s == 0.0f ? 0.0f : 1.0f/s;

        device float4 * dst4 = (device float4 *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + (ikv*G + r) + (uint64_t) iq1*args.ne1)*(D/4);

        dst4[c4] = ((threadgroup float4 *) so)[r*(D/4) + c4]*scale;
    }
}

typedef decltype(kernel_flash_attn_ext_sparse_hr<128, 4>) flash_attn_ext_sparse_hr_t;

template [[host_name("kernel_flash_attn_ext_sparse_hr_d128_nsg4")]] kernel flash_attn_ext_sparse_hr_t kernel_flash_attn_ext_sparse_hr<128, 4>;
template [[host_name("kernel_flash_attn_ext_sparse_hr_d256_nsg4")]] kernel flash_attn_ext_sparse_hr_t kernel_flash_attn_ext_sparse_hr<256, 4>;
template [[host_name("kernel_flash_attn_ext_sparse_hr_d128_nsg8")]] kernel flash_attn_ext_sparse_hr_t kernel_flash_attn_ext_sparse_hr<128, 8>;
template [[host_name("kernel_flash_attn_ext_sparse_hr_d256_nsg8")]] kernel flash_attn_ext_sparse_hr_t kernel_flash_attn_ext_sparse_hr<256, 8>;

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    constexpr short NH    = OP_LIGHTNING_INDEXER_NH;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                sq4[ih*DK4 + i4] = half4(q4[i4]);
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = ((device const float *) pw)[i_head + tiitg];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;


// ---------------------------------------------------------------------------------------------
// GGML_OP_UNION_BUILD - union-8 support.
//
// One threadgroup per BLOCK of queries. Instead of sorting 8*n_sel ids (PLAN's bitonic + 8-way
// merge), build a bitmap over the pooled rows. Scanning it in order yields a naturally ascending
// union with no sort.
//
// Membership then needs no search either - the union index of a row is its rank in the bitmap:
//     idx = wordbase[id/32] + popcount(bitmap[id/32] & ((1<<(id%32)) - 1))
// and because every query that selected a row writes the SAME id there, a single atomic_or of
// ((1<<(24+q)) | id) packs the id and the 8-bit membership mask into one word.
//
// max_union is block*n_sel, so the union can never overflow and no selection is ever dropped -
// dropping one would silently change attention output.
//
// The bitmap is a fixed 2048 words, but n_csa is NOT bounded by it: the row space is walked in
// chunks of 2048*32 rows, carrying the union offset across chunks. Chunks are visited in
// increasing row order and ids ascend within a chunk, so the union stays globally ascending.
// This is what lets the path stay on past 65536 CSA rows - it used to switch itself off there,
// silently falling back to dense attention exactly at the long contexts it was built for.
#define UNION_BUILD_MAX_WORDS 2048   // rows per chunk = 65536 (8 KB bitmap + 8 KB wordbase)

kernel void kernel_union_build(
        constant ggml_metal_kargs_union_build & args,
        device const char * sel,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg3 [[threads_per_threadgroup]]) {
    const short ntg = (short) ntg3.x;

    threadgroup atomic_uint bitmap  [UNION_BUILD_MAX_WORDS];
    threadgroup uint        wordbase[UNION_BUILD_MAX_WORDS];

    // union entries emitted by the chunks already done, and whether this chunk marked anything
    threadgroup uint         gbase;
    threadgroup atomic_uint  hit;

    const int ib = tgpig.x;

    device int32_t * out = (device int32_t *) (dst + ib*args.nb1);

    for (int i = tiitg; i < args.max_union + 1; i += ntg) {
        out[i] = 0;
    }

    if (tiitg == 0) {
        gbase = 0;
    }

    // Order device-memory zeroing before other lanes atomically OR membership into the output.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    const int n_this      = min((int) args.block, args.n_tokens - ib*args.block);
    const int chunk_rows  = UNION_BUILD_MAX_WORDS*32;

    for (int base = 0; base < args.n_csa; base += chunk_rows) {
        const int rows   = min(chunk_rows, args.n_csa - base);
        const int nwords = (rows + 31)/32;

        for (int i = tiitg; i < nwords; i += ntg) {
            atomic_store_explicit(&bitmap[i], 0u, memory_order_relaxed);
        }
        if (tiitg == 0) {
            atomic_store_explicit(&hit, 0u, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // pass A: mark every selected row that falls in this chunk
        for (int i = tiitg; i < n_this*args.n_sel; i += ntg) {
            const int q = i/args.n_sel;
            const int j = i%args.n_sel;

            device const int32_t * srow = (device const int32_t *) (sel + (ib*args.block + q)*args.nbs1);

            const int id = srow[j] - base;
            if (id >= 0 && id < rows) {
                atomic_fetch_or_explicit(&bitmap[id >> 5], 1u << (id & 31), memory_order_relaxed);
                atomic_store_explicit(&hit, 1u, memory_order_relaxed); // every lane stores the same value
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // a chunk no query selected from contributes nothing: skip its prefix sum, which is the
        // serial part, and its second pass over the selections
        if (atomic_load_explicit(&hit, memory_order_relaxed) == 0u) {
            continue;
        }

        // exclusive prefix sum of per-word popcounts, continuing from the previous chunk (serial
        // over words in one thread; it runs once per chunk per block, against an attention op that
        // is orders of magnitude larger)
        if (tiitg == 0) {
            uint acc = gbase;
            for (int w = 0; w < nwords; ++w) {
                wordbase[w] = acc;
                acc += popcount(atomic_load_explicit(&bitmap[w], memory_order_relaxed));
            }
            gbase = acc;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // pass B: rank each selection into its union slot and OR in the membership bit
        for (int i = tiitg; i < n_this*args.n_sel; i += ntg) {
            const int q = i/args.n_sel;
            const int j = i%args.n_sel;

            device const int32_t * srow = (device const int32_t *) (sel + (ib*args.block + q)*args.nbs1);

            const int id = srow[j] - base;
            if (id < 0 || id >= rows) {
                continue;
            }

            const uint w     = id >> 5;
            const uint bit   = id & 31;
            const uint below = atomic_load_explicit(&bitmap[w], memory_order_relaxed) & ((1u << bit) - 1u);
            const uint idx   = wordbase[w] + popcount(below);

            device atomic_uint * slot = (device atomic_uint *) &out[idx];
            atomic_fetch_or_explicit(slot, (1u << (24 + q)) | (uint) (id + base), memory_order_relaxed);
        }

        // the next chunk rewrites bitmap and wordbase, so pass B must be done reading them
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiitg == 0) {
        out[args.max_union] = (int32_t) gbase;
    }
}
