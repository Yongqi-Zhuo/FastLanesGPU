#include "crystal/crystal.cuh"
#include "crystal_ssb_utils.h"
#include "cub/test/test_util.h"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/hardcoded_16.cuh"
#include "fls_gen/unpack/unpack_fused.cuh"
#include "gpu_utils.h"
#include "query/query_21.hpp"
#include "ssb_utils.h"
#include "gtest/gtest.h"
#include <cub/util_allocator.cuh>
#include <iostream>
#include <vector>

#include "./benchmark.hpp"

using namespace std;
using namespace fastlanes::gpu;
using namespace fastlanes;

inline auto query_mtd = ssb::ssb_q21_100;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_v3(int* lo_orderdate,
                         int* lo_p_category,
                         int* lo_p_brand1,
                         int* lo_s_region,
                         int* lo_revenue,
                         int  lo_len,
                         int* res) {
	int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
	// Load a tile striped across threads
	int items[ITEMS_PER_THREAD];
	int selection_flags[ITEMS_PER_THREAD];
	int brand[ITEMS_PER_THREAD];
	int revenue[ITEMS_PER_THREAD];

	int tile_offset    = blockIdx.x * TILE_SIZE;
	int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
	int num_tile_items = TILE_SIZE;

	if (blockIdx.x == num_tiles - 1) { num_tile_items = lo_len - tile_offset; }

	InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

    int s_region_tile_offset = blockIdx.x * query_mtd.ssb.lo_s_chosen_region_bw * ITEMS_PER_THREAD;
    if constexpr (ITEMS_PER_THREAD == 8) {
        unpack_8_at_a_time::unpack_device(lo_s_region + s_region_tile_offset, items, query_mtd.ssb.lo_s_chosen_region_bw);
    } else {
        unpack_device(lo_s_region + s_region_tile_offset, items, query_mtd.ssb.lo_s_chosen_region_bw);
    }
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

    int p_category_tile_offset = blockIdx.x * query_mtd.ssb.lo_p_chosen_category_bw * ITEMS_PER_THREAD;
    if constexpr (ITEMS_PER_THREAD == 8) {
        unpack_8_at_a_time::unpack_device(lo_p_category + p_category_tile_offset, items, query_mtd.ssb.lo_p_chosen_category_bw);
    } else {
        unpack_device(lo_p_category + p_category_tile_offset, items, query_mtd.ssb.lo_p_chosen_category_bw);
    }
    BlockPredAndEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

	int orderdate_tile_offset = blockIdx.x * query_mtd.ssb.lo_orderdate_bw * ITEMS_PER_THREAD;
	if constexpr (ITEMS_PER_THREAD == 8) {
        unpack_8_at_a_time::unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
    } else {
        unpack_device(lo_orderdate + orderdate_tile_offset, items, query_mtd.ssb.lo_orderdate_bw);
    }
    BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        lo_p_brand1 + tile_offset, brand, num_tile_items, selection_flags);
	BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
	    lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
	for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
		if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
			if (selection_flags[ITEM]) {
                int year = (items[ITEM] + query_mtd.ssb.lo_orderdate_min) / 10000;
				int hash          = (brand[ITEM] * 7 + (year - 1992)) % ((1998 - 1992 + 1) * (5 * 5 * 40));
				res[hash * 4]     = year;
				res[hash * 4 + 1] = brand[ITEM];
				atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), (long long)(revenue[ITEM]));
			}
		}
	}
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int*                         lo_orderdate,
              int*                         lo_p_category,
              int*                         lo_p_brand1,
              int*                         lo_s_region,
              int*                         lo_revenue,
              int                          lo_len,
              cub::CachingDeviceAllocator& g_allocator,
              int                          version) {
	casdec::benchmark::Stream stream;

	int* res;
	int  res_size       = ((1998 - 1992 + 1) * (5 * 5 * 40));
	int  res_array_size = res_size * ITEMS_PER_THREAD;
	CubDebugExit(g_allocator.DeviceAllocate((void**)&res, res_array_size * sizeof(int), stream));

	auto numTotalRuns = casdec::benchmark::getDefaultNumTotalRuns();

	auto bench = casdec::benchmark::benchmarkKernel([&](int i) {
	CubDebugExit(cudaMemsetAsync(res, 0, res_array_size * sizeof(int), stream));

	int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

	if (version == 3) {
		probe_v3<BLOCK_THREADS, ITEMS_PER_THREAD><<<(lo_len + tile_items - 1) / tile_items, BLOCK_THREADS, 0, stream>>>(
		    lo_orderdate, lo_p_category, lo_p_brand1, lo_s_region, lo_revenue, lo_len, res);
	} else {
		throw std::runtime_error("this version does not exist");
	}

	}, numTotalRuns, stream);

	std::cerr << "Query time: " << bench << " ms" << std::endl;
	auto speed = lo_len / bench * 1e3;
	std::cerr << "Processing speed: " << speed << " rows/s" << std::endl;

	int* h_res = new int[res_array_size];
	CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int), cudaMemcpyDeviceToHost));

	ssb::SSBQuery2ResultTable result_of_query;
	for (int i = 0; i < res_size; i++) {
		if (h_res[4 * i] != 0) {
			result_of_query.emplace_back(
			    h_res[4 * i], h_res[4 * i + 1], reinterpret_cast<unsigned long long*>(&h_res[4 * i + 2])[0]);
		}
	}

	ASSERT_EQ(result_of_query.size(), ssb::ssb_q21_100.reuslt.size());
	ASSERT_EQ(result_of_query, ssb::ssb_q21_100.reuslt);

	delete[] h_res;

	CLEANUP(res);
}

int main(int argc, char* argv[]) {
	int version         = 3;
	// if (argc > 1) {
	// version             = std::stoi(argv[1]);
	// }
	auto hard_coded     = query_mtd.ssb;

	int* h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
	int* h_lo_p_category = loadColumn<int>("lo_p_category", LO_LEN);
    int* h_lo_p_brand1 = loadColumn<int>("lo_p_brand1", LO_LEN);
	int* h_lo_s_region = loadColumn<int>("lo_s_region", LO_LEN);
	int* h_lo_revenue   = loadColumn<int>("lo_revenue", LO_LEN);

	auto n_vec = hard_coded.n_vec;

	int* tmp = new int[n_vec * 1024];
	for (size_t i {0}; i < LO_LEN; ++i) {
		tmp[i] = h_lo_orderdate[i] - hard_coded.lo_orderdate_min;
	}

	const int* h_enc_lo_orderdate = new int[n_vec * 1024];
	const int* h_enc_lo_p_category = new int[n_vec * 1024];
	const int* h_enc_lo_s_region = new int[n_vec * 1024];

	auto* orderdate_in = const_cast<int32_t*>(tmp);
	auto* category_in = const_cast<int32_t*>(h_lo_p_category);
	auto* region_in = const_cast<int32_t*>(h_lo_s_region);

	auto* orderdate_out = const_cast<int32_t*>(h_enc_lo_orderdate);
    auto* category_out = const_cast<int32_t*>(h_enc_lo_p_category);
    auto* region_out = const_cast<int32_t*>(h_enc_lo_s_region);

	for (uint64_t vec_idx {0}; vec_idx < n_vec; vec_idx++) {
		generated::pack::fallback::scalar::pack(orderdate_in, orderdate_out, hard_coded.lo_orderdate_bw);
		orderdate_in  = orderdate_in + 1024;
		orderdate_out = orderdate_out + (hard_coded.lo_orderdate_bw * 32);

        generated::pack::fallback::scalar::pack(category_in, category_out, hard_coded.lo_p_chosen_category_bw);
        category_in  = category_in + 1024;
        category_out = category_out + (hard_coded.lo_p_chosen_category_bw * 32);

        generated::pack::fallback::scalar::pack(region_in, region_out, hard_coded.lo_s_chosen_region_bw);
        region_in  = region_in + 1024;
        region_out = region_out + (hard_coded.lo_s_chosen_region_bw * 32);
	}

	int* d_lo_orderdate = loadToGPU<int32_t>(h_enc_lo_orderdate, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_p_category = loadToGPU<int32_t>(h_enc_lo_p_category, hard_coded.n_tup_line_order, g_allocator);
    int* d_lo_p_brand1 = loadToGPU<int32_t>(h_lo_p_brand1, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_s_region = loadToGPU<int32_t>(h_enc_lo_s_region, hard_coded.n_tup_line_order, g_allocator);
	int* d_lo_revenue = loadToGPU<int32_t>(h_lo_revenue, hard_coded.n_tup_line_order, g_allocator);

	if (version == 3) {
		runQuery<32, 8>(d_lo_orderdate,
		                d_lo_p_category,
		                d_lo_p_brand1,
		                d_lo_s_region,
		                d_lo_revenue,
		                LO_LEN,
		                g_allocator,
		                version);
	} else {
		throw std::runtime_error("this version does not exist");
	}
}
