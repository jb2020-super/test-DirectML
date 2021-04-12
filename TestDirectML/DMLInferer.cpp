#include "DMLInferer.h"

using winrt::com_ptr;
using winrt::check_hresult;

DMLInferer::DMLInferer()
{
}

DMLInferer::~DMLInferer()
{
}

void DMLInferer::CreateOperator(DML_OPERATOR_TYPE op_type)
{
	m_op_type = op_type;
	switch (op_type)
	{
	case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
		CreateIdentityOp();
		break;
	case DML_OPERATOR_CONVOLUTION:
		CreateConvolutionOp();
		break;
	default:
		break;
	}
}

void DMLInferer::InitD3D12()
{
#if defined _DEBUG
	com_ptr<ID3D12Debug1> d3d12_debug;
	if (FAILED(D3D12GetDebugInterface(__uuidof(d3d12_debug), d3d12_debug.put_void()))) {
		winrt::throw_hresult(DXGI_ERROR_SDK_COMPONENT_MISSING);
	}
	else
	{
		d3d12_debug->EnableDebugLayer();
		d3d12_debug->SetEnableGPUBasedValidation(TRUE);
	}
#endif
	com_ptr<IDXGIFactory6> dxgi_factory;
	check_hresult(CreateDXGIFactory1(__uuidof(dxgi_factory), dxgi_factory.put_void()));
	com_ptr<IDXGIAdapter4> dxgi_adapter;
	// Enum GPU by performance, from low to high:
	// 1. iGPUs (integrated GPUs)
	// 2. dGPUs (discrete GPUs)
	// 3. xGPUs (external GPUs)
	check_hresult(dxgi_factory->EnumAdapterByGpuPreference(0, 
		//DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, 
		DXGI_GPU_PREFERENCE_MINIMUM_POWER,
		__uuidof(dxgi_adapter), 
		dxgi_adapter.put_void()));
	DXGI_ADAPTER_DESC3 adapter_desc{};
	check_hresult(dxgi_adapter->GetDesc3(&adapter_desc));
	check_hresult(D3D12CreateDevice(dxgi_adapter.get(), D3D_FEATURE_LEVEL_12_0, __uuidof(m_device), m_device.put_void()));
	D3D12_COMMAND_QUEUE_DESC command_queue_desc{
		D3D12_COMMAND_LIST_TYPE_DIRECT, 
		D3D12_COMMAND_QUEUE_FLAG_NONE
	};
	check_hresult(m_device->CreateCommandQueue(&command_queue_desc, __uuidof(m_cmd_queue), m_cmd_queue.put_void()));
	check_hresult(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(m_cmd_allocator), m_cmd_allocator.put_void()));
	check_hresult(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmd_allocator.get(), nullptr, __uuidof(m_cmd_list), m_cmd_list.put_void()));

}

void DMLInferer::InitDML()
{
	DML_CREATE_DEVICE_FLAGS flg = DML_CREATE_DEVICE_FLAG_NONE;
#if defined _DEBUG
	flg |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
	check_hresult(DMLCreateDevice(m_device.get(), flg, __uuidof(m_dml_device), m_dml_device.put_void()));
}

void DMLInferer::CreateConvolutionOp()
{
	UINT input_dim[4]{ 1, 1, 10, 10 };
	m_input_tensor.Create(DML_TENSOR_DATA_TYPE_FLOAT32, input_dim, 4);
	UINT filter_dim[4]{ 1, 1, 3, 3 };
	m_filter_tensor.Create(DML_TENSOR_DATA_TYPE_FLOAT32, filter_dim, 4);
	m_output_tensor.Create(DML_TENSOR_DATA_TYPE_FLOAT32, input_dim, 4);
	DML_CONVOLUTION_OPERATOR_DESC conv_op_desc{};
	conv_op_desc.InputTensor = &m_input_tensor.GetDesc();
	conv_op_desc.FilterTensor = &m_filter_tensor.GetDesc();
	conv_op_desc.BiasTensor = nullptr;
	conv_op_desc.OutputTensor = &m_output_tensor.GetDesc();
	conv_op_desc.Mode = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
	conv_op_desc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
	conv_op_desc.DimensionCount = m_input_tensor.DimCount() - 2;
	UINT strides[2] = { 1, 1 };
	conv_op_desc.Strides = strides;
	UINT dilations[2] = { 1, 1 };
	conv_op_desc.Dilations = dilations;
	UINT start_pad[2] = { 1, 1 };
	conv_op_desc.StartPadding = start_pad;
	UINT end_pad[2] = { 1, 1 };
	conv_op_desc.EndPadding = end_pad;
	UINT output_pad[2] = { 0, 0 };
	conv_op_desc.OutputPadding = output_pad;
	conv_op_desc.GroupCount = 1;
	conv_op_desc.FusedActivation = nullptr;
	DML_OPERATOR_DESC op_desc{};
	op_desc.Type = DML_OPERATOR_CONVOLUTION;
	op_desc.Desc = &conv_op_desc;
	check_hresult(m_dml_device->CreateOperator(&op_desc, __uuidof(m_dml_op), m_dml_op.put_void()));
	check_hresult(m_dml_device->CompileOperator(m_dml_op.get(), DML_EXECUTION_FLAG_NONE, __uuidof(m_pso), m_pso.put_void()));
}

void DMLInferer::CreateIdentityOp()
{
	UINT input_dim[4]{ 1, 1, 10, 10 };
	m_input_tensor.Create(DML_TENSOR_DATA_TYPE_FLOAT32, input_dim, 4);
	m_output_tensor.Create(DML_TENSOR_DATA_TYPE_FLOAT32, input_dim, 4);
	DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC op_desc{};
	op_desc.InputTensor = &m_input_tensor.GetDesc();
	op_desc.OutputTensor = &m_output_tensor.GetDesc();
	DML_OPERATOR_DESC  od{};
	od.Type = DML_OPERATOR_ELEMENT_WISE_IDENTITY;
	od.Desc = &op_desc;
	check_hresult(m_dml_device->CreateOperator(&od, __uuidof(m_dml_op), m_dml_op.put_void()));
	check_hresult(m_dml_device->CompileOperator(m_dml_op.get(), DML_EXECUTION_FLAG_NONE, __uuidof(m_pso), m_pso.put_void()));
}

void DMLInferer::InitializeOp()
{
	IDMLCompiledOperator* pso_ptrs[] = { m_pso.get() };
	check_hresult(m_dml_device->CreateOperatorInitializer(1, pso_ptrs, __uuidof(m_op_initer), m_op_initer.put_void()));

	DML_BINDING_PROPERTIES initer_prop = m_op_initer->GetBindingProperties();
	DML_BINDING_PROPERTIES op_prop = m_pso->GetBindingProperties();
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
	UINT dsp_cnt = std::max<UINT>(op_prop.RequiredDescriptorCount, initer_prop.RequiredDescriptorCount);
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	heap_desc.NumDescriptors = dsp_cnt;
	check_hresult(m_device->CreateDescriptorHeap(&heap_desc, __uuidof(m_dsp_heap), m_dsp_heap.put_void()));
	ID3D12DescriptorHeap* heap_ptrs[] = { m_dsp_heap.get() };
	m_cmd_list->SetDescriptorHeaps(1, heap_ptrs);

	DML_BINDING_TABLE_DESC tlb_desc{};
	tlb_desc.Dispatchable = m_op_initer.get();
	tlb_desc.CPUDescriptorHandle = m_dsp_heap->GetCPUDescriptorHandleForHeapStart();
	tlb_desc.GPUDescriptorHandle = m_dsp_heap->GetGPUDescriptorHandleForHeapStart();
	tlb_desc.SizeInDescriptors = dsp_cnt;
	check_hresult(m_dml_device->CreateBindingTable(&tlb_desc, __uuidof(m_binding_tlb), m_binding_tlb.put_void()));

	UINT64 tmp_size = std::max<UINT64>(initer_prop.TemporaryResourceSize, op_prop.TemporaryResourceSize);
	UINT64 pst_size = std::max<UINT64>(initer_prop.PersistentResourceSize, op_prop.PersistentResourceSize);
	if (tmp_size > 0) {
		check_hresult(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(tmp_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			__uuidof(m_tmp_buffer),
			m_tmp_buffer.put_void()
		));
		if (initer_prop.TemporaryResourceSize > 0) {
			DML_BUFFER_BINDING bd{ m_tmp_buffer.get(), 0, tmp_size };
			DML_BINDING_DESC desc{ DML_BINDING_TYPE_BUFFER, &bd };
			m_binding_tlb->BindTemporaryResource(&desc);
		}

	}
	if (pst_size > 0) {
		winrt::throw_hresult(E_NOTIMPL);
	}
	check_hresult(m_dml_device->CreateCommandRecorder(__uuidof(m_recoder), m_recoder.put_void()));
	m_recoder->RecordDispatch(m_cmd_list.get(), m_op_initer.get(), m_binding_tlb.get());

	_close_execute();
}

void DMLInferer::PrintOutput()
{
	com_ptr<ID3D12Resource> readback_buffer;
	check_hresult(m_device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(m_output_tensor.GetTensorSizeInBytes()),
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		__uuidof(readback_buffer),
		readback_buffer.put_void()));

	m_cmd_list->ResourceBarrier(1,
		&CD3DX12_RESOURCE_BARRIER::Transition(
			m_output_rc.get(),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			D3D12_RESOURCE_STATE_COPY_SOURCE
		));
	m_cmd_list->CopyResource(readback_buffer.get(), m_output_rc.get());
	_close_execute();

	D3D12_RANGE range{ 0, static_cast<SIZE_T>(m_output_tensor.GetTensorSizeInBytes()) };
	FLOAT* out_data{};
	check_hresult(readback_buffer->Map(0, &range, reinterpret_cast<void**>(&out_data)));
	
	_print_tensor(m_output_tensor, out_data);	
	
	D3D12_RANGE empty_rg{};
	readback_buffer->Unmap(0, &empty_rg);
}

void DMLInferer::ExecuteOp()
{
	ID3D12DescriptorHeap* heap_ptrs[] = { m_dsp_heap.get() };
	m_cmd_list->SetDescriptorHeaps(1, heap_ptrs);

	DML_BINDING_PROPERTIES initer_prop = m_op_initer->GetBindingProperties();
	DML_BINDING_PROPERTIES op_prop = m_pso->GetBindingProperties();
	D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
	UINT dsp_cnt = std::max<UINT>(op_prop.RequiredDescriptorCount, initer_prop.RequiredDescriptorCount);
	DML_BINDING_TABLE_DESC tlb_desc{};
	tlb_desc.Dispatchable = m_pso.get();
	tlb_desc.CPUDescriptorHandle = m_dsp_heap->GetCPUDescriptorHandleForHeapStart();
	tlb_desc.GPUDescriptorHandle = m_dsp_heap->GetGPUDescriptorHandleForHeapStart();
	tlb_desc.SizeInDescriptors = dsp_cnt;
	m_binding_tlb->Reset(&tlb_desc);

	UINT64 tmp_size = std::max<UINT64>(initer_prop.TemporaryResourceSize, op_prop.TemporaryResourceSize);
	UINT64 pst_size = std::max<UINT64>(initer_prop.PersistentResourceSize, op_prop.PersistentResourceSize);
	if (tmp_size > 0) {
		DML_BUFFER_BINDING bd{ m_tmp_buffer.get(), 0, tmp_size };
		DML_BINDING_DESC desc{ DML_BINDING_TYPE_BUFFER, &bd };
		m_binding_tlb->BindTemporaryResource(&desc);
	}
	if (pst_size > 0) {
		winrt::throw_hresult(E_NOTIMPL);
	}
	switch (m_op_type)
	{
	case DML_OPERATOR_ELEMENT_WISE_IDENTITY:
		_upload_identity_data();
		break;
	case DML_OPERATOR_CONVOLUTION:
		_upload_convolution_data();
		break;
	default:
		break;
	}

	_create_resource(nullptr, m_output_tensor.GetTensorSizeInBytes(), m_output_rc, m_upload_rc);
	DML_BUFFER_BINDING output_buffer_bd = { m_output_rc.get(), 0, m_output_rc->GetDesc().Width };
	DML_BINDING_DESC output_binding = { DML_BINDING_TYPE_BUFFER, &output_buffer_bd };
	m_binding_tlb->BindOutputs(1, &output_binding);

	m_recoder->RecordDispatch(m_cmd_list.get(), m_pso.get(), m_binding_tlb.get());
	_close_execute();
}

void DMLInferer::_create_resource(FLOAT* data, UINT64 buffer_size, winrt::com_ptr<ID3D12Resource>& rc, winrt::com_ptr<ID3D12Resource>& upload_rc)
{
	if (data) {
		check_hresult(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(buffer_size),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			__uuidof(upload_rc),
			upload_rc.put_void()
		));
		check_hresult(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			__uuidof(rc),
			rc.put_void()
		));
		D3D12_SUBRESOURCE_DATA sub_data{};
		sub_data.pData = data;
		sub_data.RowPitch = static_cast<LONG_PTR>(buffer_size);
		sub_data.SlicePitch = sub_data.RowPitch;

		UpdateSubresources(m_cmd_list.get(), rc.get(), upload_rc.get(), 0, 0, 1, &sub_data);
		m_cmd_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
			rc.get(),
			D3D12_RESOURCE_STATE_COPY_DEST,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS
		));
	}
	else
	{
		check_hresult(m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr,
			__uuidof(rc),
			rc.put_void()
		));
	}
}

void DMLInferer::_close_execute()
{
	check_hresult(m_cmd_list->Close());
	ID3D12CommandList* cmd_lst[] = { m_cmd_list.get() };
	m_cmd_queue->ExecuteCommandLists(1, cmd_lst);

	com_ptr<ID3D12Fence> fence;
	check_hresult(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(fence), fence.put_void()));
	winrt::handle fence_event{ 0 };
	fence_event.attach(::CreateEvent(nullptr, true, false, nullptr));
	winrt::check_bool(bool{ fence_event });
	check_hresult(fence->SetEventOnCompletion(1, fence_event.get()));
	check_hresult(m_cmd_queue->Signal(fence.get(), 1));
	WaitForSingleObjectEx(fence_event.get(), INFINITE, FALSE);
	check_hresult(m_cmd_allocator->Reset());
	check_hresult(m_cmd_list->Reset(m_cmd_allocator.get(), nullptr));
}

void DMLInferer::_print_tensor(DMLTensor& tensor, FLOAT* data)
{
	int height = tensor.Height();
	int width = tensor.Width();
	std::cout << "Dimension: " << height << 'x' << width << std::endl;
	for (int i = 0, idx = 0; i < tensor.Height(); ++i) {
		for (int j = 0; j < tensor.Width(); ++j) {
			std::cout << data[idx++] << '\t';
		}
		std::cout << std::endl;
	}
}

void DMLInferer::_upload_convolution_data()
{
	auto input_size = m_input_tensor.GetTensorSizeInBytes();
	std::unique_ptr<FLOAT> input_data = std::unique_ptr<FLOAT>(new FLOAT[m_input_tensor.GetElementCount()]);
	for (int i = 0; i < m_input_tensor.GetElementCount(); ++i) {
		input_data.get()[i] = 1.618f;
	}
	_print_tensor(m_input_tensor, input_data.get());
	_create_resource(input_data.get(), input_size, m_input_rc, m_upload_rc);

	auto filter_size = m_filter_tensor.GetTensorSizeInBytes();
	std::unique_ptr<FLOAT> filter_data = std::unique_ptr<FLOAT>(new FLOAT[m_filter_tensor.GetElementCount()]);
	for (int i = 0; i < m_filter_tensor.GetElementCount(); ++i) {
		filter_data.get()[i] = 1.0f;
	}
	_print_tensor(m_filter_tensor, filter_data.get());
	_create_resource(filter_data.get(), filter_size, m_filter_rc, m_upload_rc2);

	DML_BUFFER_BINDING input_buffer_bd = { m_input_rc.get(), 0, m_input_rc->GetDesc().Width };
	DML_BINDING_DESC input_binding = { DML_BINDING_TYPE_BUFFER, &input_buffer_bd };
	DML_BUFFER_BINDING filter_buffer_bd = { m_filter_rc.get(), 0, m_filter_rc->GetDesc().Width };
	DML_BINDING_DESC filter_binding = { DML_BINDING_TYPE_BUFFER, &filter_buffer_bd };
	DML_BUFFER_BINDING bias_buffer_bd = { nullptr, 0, 0 };
	DML_BINDING_DESC bias_binding = { DML_BINDING_TYPE_NONE, nullptr };

	DML_BINDING_DESC binding_descs[3] = { input_binding, filter_binding, bias_binding };

	m_binding_tlb->BindInputs(3, binding_descs);
}

void DMLInferer::_upload_identity_data()
{
	auto input_size = m_input_tensor.GetTensorSizeInBytes();
	std::unique_ptr<FLOAT> input_data = std::unique_ptr<FLOAT>(new FLOAT[m_input_tensor.GetElementCount()]);
	for (int i = 0; i < m_input_tensor.GetElementCount(); ++i) {
		input_data.get()[i] = 1.618f;
	}
	_print_tensor(m_input_tensor, input_data.get());
	_create_resource(input_data.get(), input_size, m_input_rc, m_upload_rc);
	DML_BUFFER_BINDING input_buffer_bd = { m_input_rc.get(), 0, m_input_rc->GetDesc().Width };
	DML_BINDING_DESC input_binding = { DML_BINDING_TYPE_BUFFER, &input_buffer_bd };
	DML_BINDING_DESC binding_descs[1] = { input_binding};

	m_binding_tlb->BindInputs(1, binding_descs);
}

DMLTensor::DMLTensor()
{
}

DMLTensor::~DMLTensor()
{
}

void DMLTensor::Create(DML_TENSOR_DATA_TYPE data_type, const UINT* dims, UINT dim_cnt)
{
	if (dim_cnt != 4 && dim_cnt != 5) {
		winrt::throw_hresult(E_INVALIDARG);
	}
	memcpy(m_dims, dims, sizeof(UINT) * dim_cnt);
	m_buffer_desc.DataType = data_type;
	m_buffer_desc.Flags = DML_TENSOR_FLAG_NONE;
	m_buffer_desc.DimensionCount = dim_cnt;
	m_buffer_desc.Sizes = m_dims;
	m_buffer_desc.Strides = nullptr;
	m_buffer_desc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(data_type, dim_cnt, dims, nullptr);
	m_buffer_desc.GuaranteedBaseOffsetAlignment = 256;

	m_desc.Type = DML_TENSOR_TYPE_BUFFER;
	m_desc.Desc = &m_buffer_desc;
}

DML_TENSOR_DESC& DMLTensor::GetDesc()
{
	return m_desc;
}

UINT64 DMLTensor::GetTensorSizeInBytes()
{
	return m_buffer_desc.TotalTensorSizeInBytes;
}

UINT DMLTensor::GetElementCount()
{
	UINT cnt{ 1 };
	for (int i = 0; i < m_buffer_desc.DimensionCount; ++i) {
		cnt *= m_dims[i];
	}
	return cnt;
}

UINT DMLTensor::DimCount()
{
	return 4;
}

UINT DMLTensor::Width()
{
	return m_dims[3];
}

UINT DMLTensor::Height()
{
	return m_dims[2];
}
