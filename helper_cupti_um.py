# TODO: add dropped records handling

from cupti import cupti
from cuda import cuda
from common import checkCudaErrors
import common
import atexit
import sys

um_kind_list = [
    # cupti.ActivityUnifiedMemoryCounterKind.UNKNOWN, # This is not working
    cupti.ActivityUnifiedMemoryCounterKind.BYTES_TRANSFER_HTOD,
    cupti.ActivityUnifiedMemoryCounterKind.BYTES_TRANSFER_DTOH,
    cupti.ActivityUnifiedMemoryCounterKind.CPU_PAGE_FAULT_COUNT,
    cupti.ActivityUnifiedMemoryCounterKind.GPU_PAGE_FAULT,
    cupti.ActivityUnifiedMemoryCounterKind.THRASHING,
    cupti.ActivityUnifiedMemoryCounterKind.THROTTLING,
    cupti.ActivityUnifiedMemoryCounterKind.REMOTE_MAP,
    cupti.ActivityUnifiedMemoryCounterKind.BYTES_TRANSFER_DTOD,
]

def at_exit_handler():
    cupti.activity_flush_all(1)

def setup_cupti_um(filename=sys.stdout):
    """
    Initialize CUPTI for Unified Memory profiling.
    """

    # setup file output for CUPTI activities
    if isinstance(filename, str):
        sys.stdout = open(filename, 'w')

    um_config = cupti.ActivityUnifiedMemoryCounterConfig(len(um_kind_list))

    # initialize CUDA Driver API
    checkCudaErrors(cuda.cuInit(0))
    device_count = checkCudaErrors(cuda.cuDeviceGetCount())

    if device_count < 1:
        print("No CUDA devices found.")
        sys.exit(-1)

    um_config.scope = 1 
    um_config.device_id = 0
    um_config.enable = 1
    for idx, um_kind in enumerate(um_kind_list):
        um_config[idx].kind = um_kind
    
    # Initialize CUPTI with Unified Memory configuration
    cupti_result = cupti.activity_configure_unified_memory_counter(um_config.ptr, len(um_kind_list))

    # error handling
    if cupti_result == cupti.Result.ERROR_NOT_INITIALIZED:
        print("CUPTI is not initialized. Please initialize CUPTI before configuring Unified Memory profiling.")
        sys.exit(-1)
    elif cupti_result == cupti.Result.ERROR_INVALID_PARAMETER:
        print("Invalid parameter provided for Unified Memory profiling configuration.")
        sys.exit(-1)
    elif cupti_result == cupti.Result.ERROR_UM_PROFILING_NOT_SUPPORTED:
        print("Unified Memory profiling is not supported.")
        sys.exit(-1)
    elif cupti_result == cupti.Result.ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE:
        print("Unified Memory profiling is not supported on this device.")
        sys.exit(-1)
    elif cupti_result == cupti.Result.ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES:
        print("Unified Memory profiling is not supported on non-P2P devices.")
        sys.exit(-1)
    # elif cupti_result == cupti.Result.ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS:
    #     print("Unified Memory profiling is not supported with MPS.")
    #     sys.exit(-1)
    elif cupti_result == cupti.Result.ERROR_UNKNOWN:    
        print(f"Error configuring Unified Memory profiling: {cupti_result}")
        sys.exit(-1)

    # set exit handler to flush CUPTI activities
    atexit.register(at_exit_handler)

    # initialize CUPTI
    common.cupti_initialize(
        validation=False,
    )

def free_cupti_um():
    """
    Free CUPTI resources for Unified Memory profiling.
    """
    # disable CUPTI activity
    common.cupti_activity_disable(common.default_activity_list)

    # flush CUPTI activity buffer
    common.cupti_activity_flush()

    # TODO: handling activity_get_num_dropped_records maybe needed
