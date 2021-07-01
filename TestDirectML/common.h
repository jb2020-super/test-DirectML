#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include <array>
#include <winrt/Windows.Foundation.h>
#include <Windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <DirectML.h>
#include <DXProgrammableCapture.h>
#include "d3dx12.h"

UINT64 DMLCalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    const UINT* sizes,
    const UINT* strides
);