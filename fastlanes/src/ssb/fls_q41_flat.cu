// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include "crystal/crystal.cuh"
#include "crystal_ssb_utils.h"
#include "cub/test/test_util.h"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/hardcoded_16.cuh"
#include "fls_gen/unpack/unpack_fused.cuh"
#include "gpu_utils.h"
#include "query/query_41.hpp"
#include "ssb_utils.h"
#include "gtest/gtest.h"
#include <cub/util_allocator.cuh>
#include <iostream>
#include <vector>

#include "./benchmark.hpp"

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

inline auto query_mtd = ssb::ssb_q41_100;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_v2(int* lo_orderdate,
                         int* lo_c_region,
                         int* lo_s_region,
                         int* lo_p_mfgr,
                         int* lo_c_nation,
                         int* lo_revenue,
                         int* lo_supplycost,
                         int  lo_len,
                         int* res) {
	constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	int years[ITEMS_PER_THREAD];
	int c_nation[ITEMS_PER_THREAD];
	int revenue[ITEMS_PER_THREAD];
	int supplycost[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;
	if (blockIdx.x == num_tiles - 1) { num_tile_items = lo_len - tile_offset; }

	if (num_tile_items <= 0) { return; }

	InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

	int c_region_tile_offset = blockIdx.x * query_mtd.ssb.lo_c_chosen_region_bw * ITEMS_PER_THREAD;
	if constexpr (ITEMS_PER_THREAD == 8) {
		unpack_8_at_a_time::unpack_device(lo_c_region + c_region_tile_offset,
		                                 items,
		                                 query_mtd.ssb.lo_c_chosen_region_bw);
	} else {
		unpack_device(lo_c_region + c_region_tile_offset, items, query_mtd.ssb.lo_c_chosen_region_bw);
	}
	BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

	int s_region_tile_offset = blockIdx.x * query_mtd.ssb.lo_s_chosen_region_bw * ITEMS_PER_THREAD;
	if constexpr (ITEMS_PER_THREAD == 8) {
		unpack_8_at_a_time::unpack_device(lo_s_region + s_region_tile_offset,
		                                 items,
		                                 query_mtd.ssb.lo_s_chosen_region_bw);
	} else {
		unpack_device(lo_s_region + s_region_tile_offset, items, query_mtd.ssb.lo_s_chosen_region_bw);
	}
	BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

	int p_mfgr_tile_offset = blockIdx.x * query_mtd.ssb.lo_p_chosen_mfgr_bw * ITEMS_PER_THREAD;
	if constexpr (ITEMS_PER_THREAD == 8) {
		unpack_8_at_a_time::unpack_device(lo_p_mfgr + p_mfgr_tile_offset,
		                                 items,
		                                 query_mtd.ssb.lo_p_chosen_mfgr_bw);
	} else {
		unpack_device(lo_p_mfgr + p_mfgr_tile_offset, items, query_mtd.ssb.lo_p_chosen_mfgr_bw);
	}
	#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		int logical_idx = threadIdx.x + (BLOCK_THREADS * ITEM);
		if (logical_idx < num_tile_items) {
			if (!(items[ITEM] == 0 || items[ITEM] == 1)) { selection_flags[ITEM] = 0; }
		}
	}

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * ITEMS_PER_THREAD;
	if constexpr (ITEMS_PER_THREAD == 8) {
		unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset,
		                                 items,
		                                 query_mtd.ssb.lo_orderdate_bw);
	} else {
		unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
	}
	#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		int logical_idx = threadIdx.x + (BLOCK_THREADS * ITEM);
		if (logical_idx < num_tile_items) {
			years[ITEM] = (items[ITEM] + query_mtd.ssb.lo_orderdate_min) / 10000;
		}
	}

	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_c_nation + tile_offset, c_nation, num_tile_items, selection_flags);
	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);
	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_supplycost + tile_offset, supplycost, num_tile_items, selection_flags);

	#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		int logical_idx = threadIdx.x + (BLOCK_THREADS * ITEM);
		if (logical_idx < num_tile_items && selection_flags[ITEM]) {
			int hash = (c_nation[ITEM] * 7 + (years[ITEM] - 1992)) % ((1998 - 1992 + 1) * 25);
			res[hash * 4]     = years[ITEM];
			res[hash * 4 + 1] = c_nation[ITEM];
			atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]),
			          (long long)(revenue[ITEM] - supplycost[ITEM]));
		}
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int*                         lo_orderdate,
              int*                         lo_c_region,
              int*                         lo_s_region,
              int*                         lo_p_mfgr,
              int*                         lo_c_nation,
              int*                         lo_revenue,
              int*                         lo_supplycost,
              int                          lo_len,
              cub::CachingDeviceAllocator& g_allocator,
              int                          version) {
	casdec::benchmark::Stream stream;

	int* res;
	int  res_size       = ((1998 - 1992 + 1) * 25);
	int  res_array_size = res_size * 4;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int), stream));

	auto numTotalRuns = casdec::benchmark::getDefaultNumTotalRuns();

	auto bench = casdec::benchmark::benchmarkKernel([&](int) {
	CubDebugExit(cudaMemsetAsync(res, 0, res_array_size * sizeof(int), stream));

	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
	if (version == 2) {
		probe_v2<BLOCK_THREADS, ITEMS_PER_THREAD><<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS, 0, stream>>>(
		    lo_orderdate,
		    lo_c_region,
		    lo_s_region,
		    lo_p_mfgr,
		    lo_c_nation,
		    lo_revenue,
		    lo_supplycost,
		    lo_len,
		    res);
	} else {
		throw std::runtime_error("this version does not exist");
	}

	}, numTotalRuns, stream);

	std::cerr << "Query time: " << bench << " ms" << std::endl;
	auto speed = lo_len / bench * 1e3;
	std::cerr << "Processing speed: " << speed << " rows/s" << std::endl;

	int* h_res = new int[res_array_size];
	CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));

	ssb::SSBQuery4ResultTable result_of_query;
	for (int i = 0; i < res_size; i++) {
		if (h_res[4 * i] != 0) {
			result_of_query.emplace_back(
			    h_res[4 * i], h_res[4 * i + 1], reinterpret_cast<unsigned long long*>(&h_res[4 * i + 2])[0]);
		}
	}

	ASSERT_EQ(result_of_query.size(), query_mtd.reuslt.size());
	ASSERT_EQ(result_of_query, query_mtd.reuslt);

	delete[] h_res;

	CLEANUP(res);
}

int main(int argc, char* argv[]) {
	int version = 2;
	if (argc > 1) {
		version = std::stoi(argv[1]);
	}

	auto hard_coded      = query_mtd.ssb;
	int* h_lo_orderdate  = loadColumn<int>("lo_orderdate", LO_LEN);
	int* h_lo_c_region   = loadColumn<int>("lo_c_region", LO_LEN);
	int* h_lo_s_region   = loadColumn<int>("lo_s_region", LO_LEN);
	int* h_lo_p_mfgr     = loadColumn<int>("lo_p_mfgr", LO_LEN);
	int* h_lo_c_nation   = loadColumn<int>("lo_c_nation", LO_LEN);
	int* h_lo_revenue    = loadColumn<int>("lo_revenue", LO_LEN);
	int* h_lo_supplycost = loadColumn<int>("lo_supplycost", LO_LEN);

	auto n_vec = hard_coded.n_vec;

	int* tmp = new int[n_vec * 1024];
	for (size_t i {0}; i < LO_LEN; ++i) {
		tmp[i] = h_lo_orderdate[i] - hard_coded.lo_orderdate_min;
	}

	const int* h_enc_lo_orderdate = new int[n_vec * 1024];
	const int* h_enc_lo_c_region  = new int[n_vec * 1024];
	const int* h_enc_lo_s_region  = new int[n_vec * 1024];
	const int* h_enc_lo_p_mfgr    = new int[n_vec * 1024];

	auto* orderdate_in = const_cast<int32_t*>(tmp);
	auto* c_region_in  = const_cast<int32_t*>(h_lo_c_region);
	auto* s_region_in  = const_cast<int32_t*>(h_lo_s_region);
	auto* p_mfgr_in    = const_cast<int32_t*>(h_lo_p_mfgr);

	auto* orderdate_out = const_cast<int32_t*>(h_enc_lo_orderdate);
	auto* c_region_out  = const_cast<int32_t*>(h_enc_lo_c_region);
	auto* s_region_out  = const_cast<int32_t*>(h_enc_lo_s_region);
	auto* p_mfgr_out    = const_cast<int32_t*>(h_enc_lo_p_mfgr);

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, hard_coded.lo_orderdate_bw);
		orderdate_in  = orderdate_in + 1024;
		orderdate_out = orderdate_out + (hard_coded.lo_orderdate_bw * 32);

		generated::pack::fallback::scalar::pack(c_region_in, c_region_out, hard_coded.lo_c_chosen_region_bw);
		c_region_in  = c_region_in + 1024;
		c_region_out = c_region_out + (hard_coded.lo_c_chosen_region_bw * 32);

		generated::pack::fallback::scalar::pack(s_region_in, s_region_out, hard_coded.lo_s_chosen_region_bw);
		s_region_in  = s_region_in + 1024;
		s_region_out = s_region_out + (hard_coded.lo_s_chosen_region_bw * 32);

		generated::pack::fallback::scalar::pack(p_mfgr_in, p_mfgr_out, hard_coded.lo_p_chosen_mfgr_bw);
		p_mfgr_in  = p_mfgr_in + 1024;
		p_mfgr_out = p_mfgr_out + (hard_coded.lo_p_chosen_mfgr_bw * 32);
	}

	int* d_lo_orderdate  = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order * hard_coded.lo_orderdate_bw / 32, g_allocator);
	int* d_lo_c_region   = loadToGPU<int32_t>(h_enc_lo_c_region, hard_coded.n_tup_line_order * hard_coded.lo_c_chosen_region_bw / 32, g_allocator);
	int* d_lo_s_region   = loadToGPU<int32_t>(h_enc_lo_s_region, hard_coded.n_tup_line_order * hard_coded.lo_s_chosen_region_bw / 32, g_allocator);
	int* d_lo_p_mfgr     = loadToGPU<int32_t>(h_enc_lo_p_mfgr, hard_coded.n_tup_line_order * hard_coded.lo_p_chosen_mfgr_bw / 32, g_allocator);
	int* d_lo_c_nation   = loadToGPU<int32_t>(h_lo_c_nation, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_revenue    = loadToGPU<int32_t>(h_lo_revenue, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_supplycost = loadToGPU<int32_t>(h_lo_supplycost, hard_coded.n_tup_line_order, g_allocator);

	runQuery<32, 8>(d_lo_orderdate,
	               d_lo_c_region,
	               d_lo_s_region,
	               d_lo_p_mfgr,
	               d_lo_c_nation,
	               d_lo_revenue,
	               d_lo_supplycost,
	               LO_LEN,
	               g_allocator,
	               version);

	return 0;
}
