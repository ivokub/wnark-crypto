package bn254

import (
	"encoding/binary"
	"fmt"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	_ "github.com/gogpu/wgpu/hal/allbackends"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

type FrOp uint32

const (
	FrOpCopy FrOp = iota
	FrOpZero
	FrOpOne
	FrOpAdd
	FrOpSub
	FrOpNeg
	FrOpDouble
	FrOpNormalize
	FrOpEqual
	FrOpMul
	FrOpSquare
	FrOpToMont
	FrOpFromMont
)

const (
	frElementBytes = 32
	frUniformBytes = 16
	frWorkgroup    = 64
)

type DeviceSet struct {
	Instance *wgpu.Instance
	Adapter  *wgpu.Adapter
	Device   *wgpu.Device
}

type FrKernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
}

func NewHeadlessDevice() (*DeviceSet, error) {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("CreateInstance: %w", err)
	}
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("RequestAdapter: %w", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("RequestDevice: %w", err)
	}
	return &DeviceSet{
		Instance: instance,
		Adapter:  adapter,
		Device:   device,
	}, nil
}

func (d *DeviceSet) Close() {
	if d == nil {
		return
	}
	if d.Device != nil {
		d.Device.Release()
	}
	if d.Adapter != nil {
		d.Adapter.Release()
	}
	if d.Instance != nil {
		d.Instance.Release()
	}
}

func ReadFrArithShader() (string, error) {
	return curvegpu.ReadShader("curves/bn254/fr_arith.wgsl")
}

func ZeroBatch(count int) []curvegpu.U32x8 {
	return make([]curvegpu.U32x8, count)
}

func NewFrKernel(device *wgpu.Device) (*FrKernel, error) {
	shaderText, err := ReadFrArithShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bn254-fr-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bn254-fr-bgl",
		Entries: []wgpu.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 2, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
			{Binding: 3, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		},
	})
	if err != nil {
		shader.Release()
		return nil, fmt.Errorf("create bind group layout: %w", err)
	}
	pipelineLayout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		Label:            "bn254-fr-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bn254-fr-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "fr_ops_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	return &FrKernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
	}, nil
}

func (k *FrKernel) Close() {
	if k == nil {
		return
	}
	if k.pipeline != nil {
		k.pipeline.Release()
	}
	if k.pipelineLayout != nil {
		k.pipelineLayout.Release()
	}
	if k.layout != nil {
		k.layout.Release()
	}
	if k.shader != nil {
		k.shader.Release()
	}
}

func RunFrOperation(device *wgpu.Device, op FrOp, a, b []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	kernel, err := NewFrKernel(device)
	if err != nil {
		return nil, err
	}
	defer kernel.Close()
	return kernel.Run(op, a, b)
}

func (k *FrKernel) Run(op FrOp, a, b []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	count, err := inferCount(a, b)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}

	inA := normalizeBatch(a, count)
	inB := normalizeBatch(b, count)

	inSize := uint64(count * frElementBytes)
	outSize := inSize

	inputA, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-input-a",
		Size:  inSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_a buffer: %w", err)
	}
	defer inputA.Release()

	inputB, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-input-b",
		Size:  inSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create input_b buffer: %w", err)
	}
	defer inputB.Release()

	output, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-output",
		Size:  outSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("create output buffer: %w", err)
	}
	defer output.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-staging",
		Size:  outSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-params",
		Size:  frUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	if err := k.device.Queue().WriteBuffer(inputA, 0, flattenBatch(inA)); err != nil {
		return nil, fmt.Errorf("write input_a buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(inputB, 0, flattenBatch(inB)); err != nil {
		return nil, fmt.Errorf("write input_b buffer: %w", err)
	}

	params := make([]byte, frUniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(op))
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, fmt.Errorf("write uniform buffer: %w", err)
	}

	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bn254-fr-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputA, Size: inSize},
			{Binding: 1, Buffer: inputB, Size: inSize},
			{Binding: 2, Buffer: output, Size: outSize},
			{Binding: 3, Buffer: uniform, Size: frUniformBytes},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create bind group: %w", err)
	}
	defer bindGroup.Release()

	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("create command encoder: %w", err)
	}

	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return nil, fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32((count+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(output, 0, staging, 0, outSize)

	cmd, err := encoder.Finish()
	if err != nil {
		return nil, fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, fmt.Errorf("submit: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, fmt.Errorf("wait idle: %w", err)
	}

	data := make([]byte, outSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, fmt.Errorf("read output buffer: %w", err)
	}
	return unflattenBatch(data, count), nil
}

func (k *FrKernel) Copy(a []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpCopy, a, ZeroBatch(len(a)))
}

func (k *FrKernel) Add(a, b []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpAdd, a, b)
}

func (k *FrKernel) Sub(a, b []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpSub, a, b)
}

func (k *FrKernel) Mul(a, b []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpMul, a, b)
}

func (k *FrKernel) ToMont(a []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpToMont, a, ZeroBatch(len(a)))
}

func (k *FrKernel) FromMont(a []curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	return k.Run(FrOpFromMont, a, ZeroBatch(len(a)))
}

func inferCount(a, b []curvegpu.U32x8) (int, error) {
	switch {
	case len(a) == 0 && len(b) == 0:
		return 0, nil
	case len(a) == 0:
		return len(b), nil
	case len(b) == 0:
		return len(a), nil
	case len(a) != len(b):
		return 0, fmt.Errorf("input length mismatch: %d != %d", len(a), len(b))
	default:
		return len(a), nil
	}
}

func normalizeBatch(in []curvegpu.U32x8, count int) []curvegpu.U32x8 {
	if len(in) == count {
		return in
	}
	out := make([]curvegpu.U32x8, count)
	copy(out, in)
	return out
}

func flattenBatch(in []curvegpu.U32x8) []byte {
	out := make([]byte, len(in)*frElementBytes)
	for i, element := range in {
		base := i * frElementBytes
		for j, limb := range element {
			binary.LittleEndian.PutUint32(out[base+j*4:], limb)
		}
	}
	return out
}

func unflattenBatch(data []byte, count int) []curvegpu.U32x8 {
	out := make([]curvegpu.U32x8, count)
	for i := 0; i < count; i++ {
		base := i * frElementBytes
		for j := range out[i] {
			out[i][j] = binary.LittleEndian.Uint32(data[base+j*4:])
		}
	}
	return out
}
