//==============================================================================
// Auto Generated Code for AttentionOpPackage
//==============================================================================
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

// Kernel function type — must match the signature from hexagon-nm output
typedef void (*TritonAttentionKernelFunc)(
    const void*, const void*, const void*, void*);

// Defined in AttentionOpPackageInterface.cpp, loaded at init time
extern void* g_triton_handle;
extern TritonAttentionKernelFunc g_run_triton_attention;

BEGIN_PKG_OP_DEFINITION(PKG_AttentionFwdOp);

template<typename TensorType>
GraphStatus attentionfwdopImpl(TensorType& out_0,
                               const TensorType& query,
                               const TensorType& key,
                               const TensorType& value)
{
    if (!g_run_triton_attention) return GraphStatus::ErrorFatal;

    // raw_data_const() / raw_data() — verified accessor for QNN HTP Tensor
    // If these don't compile, check:
    //   grep -rn "raw_data\|get_data\|getData" $QNN_SDK_ROOT/include/HTP/core/
    const void* query_ptr = query.raw_data_const();
    const void* key_ptr   = key.raw_data_const();
    const void* value_ptr = value.raw_data_const();
    void* out_ptr         = out_0.raw_data();

    if (!query_ptr || !key_ptr || !value_ptr || !out_ptr)
        return GraphStatus::ErrorFatal;

    g_run_triton_attention(query_ptr, key_ptr, value_ptr, out_ptr);
    return GraphStatus::Success;
}

DEF_PACKAGE_OP((attentionfwdopImpl<Tensor>), "AttentionFwdOp")

__attribute__((unused)) static float attentionfwdopCostFunc(const Op* op) {
    return 10000.0f;
}

END_PKG_OP_DEFINITION(PKG_AttentionFwdOp);
