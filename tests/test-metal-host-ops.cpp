// Metal host ops (on unless GGML_METAL_HOST_OPS=0): map_custom nodes marked with ggml_map_custom_set_host_op run on
// a host thread inside the Metal graph. Checks that the scheduler keeps the graph in one split and that
// every host op sees its producer's output and hands its own to the next GPU op, over many layers
// spread across several command buffers and repeated graph computes, and that an async compute is safe
// when the caller frees the graph right away (llama_decode does not wait for the graph it built).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

static constexpr int N_LAYERS = 48;
static constexpr int N_ELEM   = 256;

// dst = src + (layer + 1), on the host
static void host_add_layer(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    CHECK(ith == 0 && nth == 1);
    const float k = (float) (*(const int *) userdata + 1);
    const float * x = (const float *) a->data;
    float * y = (float *) dst->data;
    for (int64_t i = 0; i < ggml_nelements(dst); ++i) {
        y[i] = x[i] + k;
    }
}

static int layer_ids[N_LAYERS];

static ggml_cgraph * build_graph(ggml_context * ctx, ggml_tensor ** px, ggml_tensor ** pone, ggml_tensor ** py) {
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEM);
    ggml_set_input(x);
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEM);
    ggml_set_input(one);

    ggml_tensor * y = x;
    ggml_tensor * z = x; // independent of the host ops: may run while the host works
    for (int l = 0; l < N_LAYERS; ++l) {
        layer_ids[l] = l;
        // a few GPU ops per layer so the graph spans several command buffers
        for (int k = 0; k < 4; ++k) {
            y = ggml_add(ctx, y, one);
        }
        y = ggml_map_custom1(ctx, y, host_add_layer, 1, &layer_ids[l]);
        ggml_map_custom_set_host_op(y);
        for (int k = 0; k < 3; ++k) {
            z = ggml_add(ctx, z, one);
        }
        y = ggml_scale(ctx, y, 1.0f);
    }
    y = ggml_add(ctx, y, z);
    ggml_set_output(y);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, y);

    *px = x; *pone = one; *py = y;
    return gf;
}

// no weights pull this graph onto the GPU, so place every node there, as a model's would be
static void pin_to_gpu(ggml_backend_sched_t sched, ggml_cgraph * gf, ggml_backend_t gpu) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(gf, i), gpu);
    }
}

// y: each layer adds 4 on the GPU and (l + 1) on the host; z: 3 per layer on the GPU
static void check_out(const std::vector<float> & in, const std::vector<float> & out) {
    const float add = 4.0f*N_LAYERS + (float) (N_LAYERS*(N_LAYERS + 1)/2);
    for (int i = 0; i < N_ELEM; ++i) {
        CHECK(out[i] == (in[i] + add) + (in[i] + 3.0f*N_LAYERS));
    }
}

int main() {
    setenv("GGML_METAL_HOST_OPS", "1", 1);

    try {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
        if (!dev) { dev = ggml_backend_dev_by_name("Metal"); }
        if (!dev) { puts("Metal unavailable: SKIP"); return 77; }

        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_t backends[2] = { gpu, cpu };

        const size_t mem_size = ggml_tensor_overhead()*(N_LAYERS*16 + 16) + ggml_graph_overhead_custom(4096, false);
        ggml_context * ctx = ggml_init({ mem_size, nullptr, true });

        ggml_tensor * x, * one, * y;
        ggml_cgraph * gf = build_graph(ctx, &x, &one, &y);

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, true);
        pin_to_gpu(sched, gf, gpu);
        CHECK(ggml_backend_sched_alloc_graph(sched, gf));

        std::vector<float> ones(N_ELEM, 1.0f), in(N_ELEM), out(N_ELEM);
        ggml_backend_tensor_set(one, ones.data(), 0, sizeof(float)*N_ELEM);

        for (int rep = 0; rep < 50; ++rep) {
            for (int i = 0; i < N_ELEM; ++i) { in[i] = (float) (i + rep); }
            ggml_backend_tensor_set(x, in.data(), 0, sizeof(float)*N_ELEM);

            CHECK(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_sched_get_n_splits(sched) == 1);

            ggml_backend_tensor_get(y, out.data(), 0, sizeof(float)*N_ELEM);
            check_out(in, out);
        }
        ggml_free(ctx);

        // the host ops read their graph nodes: poison and free the graph as soon as the async compute returns
        std::vector<uint8_t> mem(mem_size);
        for (int rep = 0; rep < 20; ++rep) {
            ctx = ggml_init({ mem_size, mem.data(), true });
            gf = build_graph(ctx, &x, &one, &y);
            ggml_backend_sched_reset(sched);
            pin_to_gpu(sched, gf, gpu);
            CHECK(ggml_backend_sched_alloc_graph(sched, gf));

            for (int i = 0; i < N_ELEM; ++i) { in[i] = (float) (i - rep); }
            ggml_backend_tensor_set(one, ones.data(), 0, sizeof(float)*N_ELEM);
            ggml_backend_tensor_set(x, in.data(), 0, sizeof(float)*N_ELEM);

            CHECK(ggml_backend_sched_graph_compute_async(sched, gf) == GGML_STATUS_SUCCESS);
            const ggml_tensor y_copy = *y;
            memset(mem.data(), 0xff, mem.size());
            ggml_free(ctx);

            ggml_backend_sched_synchronize(sched);
            ggml_backend_tensor_get(&y_copy, out.data(), 0, sizeof(float)*N_ELEM);
            check_out(in, out);
        }

        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_backend_free(gpu);

        puts("metal host ops: OK");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
