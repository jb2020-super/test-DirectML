#pragma once
#include "common.h"

class DMLTensor 
{
public:
	DMLTensor();
	~DMLTensor();
	void Create(DML_TENSOR_DATA_TYPE data_type, const UINT* dims, UINT dim_cnt);
	DML_TENSOR_DESC& GetDesc();
	UINT DimCount();
private:
	UINT m_dims[4];
	DML_BUFFER_TENSOR_DESC m_buffer_desc;
	DML_TENSOR_DESC m_desc;
};

class DMLInferer
{
public:
	DMLInferer();
	~DMLInferer();
	void InitD3D12();
	void InitDML();
	void CreateConvolutionOp();
	void InitializeOp();
	void ExecuteOp();
private:
	winrt::com_ptr<ID3D12Device8> m_device;
	winrt::com_ptr<ID3D12CommandQueue> m_cmd_queue;
	winrt::com_ptr<ID3D12CommandAllocator> m_cmd_allocator;
	winrt::com_ptr<ID3D12GraphicsCommandList> m_cmd_list;
	winrt::com_ptr<IDMLDevice> m_dml_device;
	winrt::com_ptr<IDMLOperator> m_dml_op;
	winrt::com_ptr<IDMLCompiledOperator> m_pso;
	winrt::com_ptr<IDMLOperatorInitializer> m_op_initer;
	winrt::com_ptr<ID3D12DescriptorHeap> m_dsp_heap;
	winrt::com_ptr<IDMLBindingTable> m_binding_tlb;
	winrt::com_ptr<IDMLCommandRecorder> m_recoder;
};

