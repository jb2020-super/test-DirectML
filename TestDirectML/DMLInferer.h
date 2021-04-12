#pragma once
#include "common.h"

class DMLTensor 
{
public:
	DMLTensor();
	~DMLTensor();
	void Create(DML_TENSOR_DATA_TYPE data_type, const UINT* dims, UINT dim_cnt);
	
	DML_TENSOR_DESC& GetDesc();
	UINT64 GetTensorSizeInBytes();
	UINT GetElementCount();
	UINT DimCount();
	UINT Width();
	UINT Height();

private:
	
	UINT m_dims[5];
	DML_BUFFER_TENSOR_DESC m_buffer_desc;
	DML_TENSOR_DESC m_desc;
};

class DMLInferer
{
public:
	DMLInferer();
	~DMLInferer();
	void CreateOperator(DML_OPERATOR_TYPE op_type);
	void CreateConvolutionOp();
	void CreateIdentityOp();
	void ExecuteOp();
	void InitD3D12();
	void InitDML();
	void InitializeOp();
	void PrintOutput();
private:
	void _create_resource(FLOAT* data, UINT64 buffer_size, winrt::com_ptr<ID3D12Resource>& rc, winrt::com_ptr<ID3D12Resource>& upload_rc);
	void _close_execute();
	void _print_tensor(DMLTensor& tensor, FLOAT* data);
	void _upload_convolution_data();
	void _upload_identity_data();


	DML_OPERATOR_TYPE m_op_type{};
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
	winrt::com_ptr<ID3D12Resource> m_input_rc;
	winrt::com_ptr<ID3D12Resource> m_filter_rc;
	winrt::com_ptr<ID3D12Resource> m_output_rc;
	winrt::com_ptr<ID3D12Resource> m_tmp_buffer;
	winrt::com_ptr<ID3D12Resource> m_upload_rc;
	winrt::com_ptr<ID3D12Resource> m_upload_rc2;
	DMLTensor m_input_tensor;
	DMLTensor m_filter_tensor;
	DMLTensor m_output_tensor;
};

