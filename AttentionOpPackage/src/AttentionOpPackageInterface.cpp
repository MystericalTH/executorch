//==============================================================================
// Auto Generated Code for AttentionOpPackage
//==============================================================================

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include <dlfcn.h>
#include <string>

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_AttentionFwdOp)
END_PKG_OPS_OPTS_LIST()

typedef void (*TritonAttentionKernelFunc)(
    const void*, const void*, const void*, void*);

// op package info
static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static std::array<const char*, 1> sg_opNames{{"AttentionFwdOp"}};
static Qnn_ApiVersion_t sg_sdkApiVersion  = QNN_HTP_API_VERSION_INIT;
static QnnOpPackage_Info_t sg_packageInfo = QNN_OP_PACKAGE_INFO_INIT;
static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_packageInitialized = false;

// Logging
static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel    = (QnnLog_Level_t)0;
static bool sg_logInitialized           = false;

// ── Triton kernel globals ──────────────────────────────────────────────────
// Shared with AttentionFwdOp.cpp via extern declarations in that file
void* g_triton_handle                      = nullptr;
TritonAttentionKernelFunc g_run_triton_attention = nullptr;
// ──────────────────────────────────────────────────────────────────────────

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()
INIT_PACKAGE_PARAM_ORDER_DEF()
INIT_PKG_CORE_INIT_FUNC()

static std::string resolveKernelPath(const char* kernelSoName) {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&g_triton_handle), &info) == 0 ||
        !info.dli_fname) {
        return std::string();
    }

    std::string package_path(info.dli_fname);
    const auto slash = package_path.find_last_of('/');
    if (slash == std::string::npos) {
        return std::string(kernelSoName);
    }
    return package_path.substr(0, slash + 1) + kernelSoName;
}

Qnn_ErrorHandle_t AttentionOpPackageInit(
    QnnOpPackage_GlobalInfrastructure_t infrastructure)
{
    if (sg_packageInitialized)
        return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;

    // ── Load the Triton kernel .so once at package init ──────────────────
    const std::string kernel_path = resolveKernelPath("libattention_fwd_kernel.so");
    if (kernel_path.empty()) return QNN_OP_PACKAGE_ERROR_GENERAL;

    g_triton_handle = dlopen(kernel_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_triton_handle) return QNN_OP_PACKAGE_ERROR_GENERAL;

    g_run_triton_attention = (TritonAttentionKernelFunc)
        dlsym(g_triton_handle, "attention_fwd_kernel");
    if (!g_run_triton_attention) {
        dlclose(g_triton_handle);
        g_triton_handle = nullptr;
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    }
    // ─────────────────────────────────────────────────────────────────────

    REGISTER_PACKAGE_PARAM_ORDERS()
    REGISTER_PACKAGE_AXIS_PARAMS()
    REGISTER_PACKAGE_PER_CHANNEL_QUANTIZED_OPS()

    sg_globalInfra        = infrastructure;
    sg_packageInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
    if (!sg_packageInitialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

    sg_packageInfo                = QNN_OP_PACKAGE_INFO_INIT;
    sg_packageInfo.packageName    = sg_packageName;
    sg_packageInfo.operationNames = sg_opNames.data();
    sg_packageInfo.numOperations  = sg_opNames.size();
    sg_packageInfo.sdkBuildId     = QNN_SDK_BUILD_ID;
    sg_packageInfo.sdkApiVersion  = &sg_sdkApiVersion;

    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageLogInitialize(
    QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel)
{
    if (sg_logInitialized)
        return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    if (!callback) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback    = callback;
    sg_maxLogLevel    = maxLogLevel;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageLogSetLevel(QnnLog_Level_t maxLogLevel) {
    if (maxLogLevel < QNN_LOG_LEVEL_ERROR) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_maxLogLevel = maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageLogTerminate() {
    if (!sg_logInitialized)
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_logCallback    = nullptr;
    sg_maxLogLevel    = (QnnLog_Level_t)0;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (std::string(sg_packageName) != opConfig.v1.packageName)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;

    if (std::string(opConfig.v1.typeName) == "AttentionFwdOp") {
        if (opConfig.v1.numOfParams   != 0 ||
            opConfig.v1.numOfInputs   != 3 ||
            opConfig.v1.numOfOutputs  != 1)
            return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    } else {
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    }
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t AttentionOpPackageTerminate() {
    if (!sg_packageInitialized)
        return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;

    // ── Unload the Triton kernel .so ─────────────────────────────────────
    if (g_triton_handle) {
        dlclose(g_triton_handle);
        g_triton_handle             = nullptr;
        g_run_triton_attention      = nullptr;
    }
    // ─────────────────────────────────────────────────────────────────────

    sg_globalInfra        = nullptr;
    sg_packageInitialized = false;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t AttentionOpPackageInterfaceProvider(
    QnnOpPackage_Interface_t* interface)
{
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion      = {1, 4, 0};
    interface->v1_4.init             = AttentionOpPackageInit;
    interface->v1_4.terminate        = AttentionOpPackageTerminate;
    interface->v1_4.getInfo          = AttentionOpPackageGetInfo;
    interface->v1_4.validateOpConfig = AttentionOpPackageValidateOpConfig;
    interface->v1_4.createOpImpl     = nullptr;
    interface->v1_4.freeOpImpl       = nullptr;
    interface->v1_4.logInitialize    = AttentionOpPackageLogInitialize;
    interface->v1_4.logSetLevel      = AttentionOpPackageLogSetLevel;
    interface->v1_4.logTerminate     = AttentionOpPackageLogTerminate;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
