package bn254

import (
	"encoding/binary"
	"fmt"
	"time"

	"github.com/gogpu/gputypes"
	"github.com/gogpu/wgpu"

	"github.com/ivokub/wnark-crypto/go/curvegpu"
)

const frNTTUniformBytes = 16

type FrNTTKernel struct {
	device         *wgpu.Device
	shader         *wgpu.ShaderModule
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
	pipeline       *wgpu.ComputePipeline
	vectorKernel   *FrVectorKernel
}

type FrNTTStageProfile struct {
	Upload   time.Duration
	Kernel   time.Duration
	Readback time.Duration
	Total    time.Duration
}

type FrNTTProfile struct {
	BitReverse FrVectorProfile
	Stages     FrNTTStageProfile
	Scale      FrVectorProfile
	Total      time.Duration
}

func ReadFrNTTShader() (string, error) {
	return curvegpu.ReadShader("curves/bn254/fr_ntt.wgsl")
}

func NewFrNTTKernel(device *wgpu.Device) (*FrNTTKernel, error) {
	shaderText, err := ReadFrNTTShader()
	if err != nil {
		return nil, err
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "bn254-fr-ntt-shader",
		WGSL:  shaderText,
	})
	if err != nil {
		return nil, fmt.Errorf("create shader module: %w", err)
	}
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "bn254-fr-ntt-bgl",
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
		Label:            "bn254-fr-ntt-pl",
		BindGroupLayouts: []*wgpu.BindGroupLayout{layout},
	})
	if err != nil {
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:      "bn254-fr-ntt-pipeline",
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "fr_ntt_stage_main",
	})
	if err != nil {
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	vectorKernel, err := NewFrVectorKernel(device)
	if err != nil {
		pipeline.Release()
		pipelineLayout.Release()
		layout.Release()
		shader.Release()
		return nil, fmt.Errorf("create vector kernel: %w", err)
	}
	return &FrNTTKernel{
		device:         device,
		shader:         shader,
		layout:         layout,
		pipelineLayout: pipelineLayout,
		pipeline:       pipeline,
		vectorKernel:   vectorKernel,
	}, nil
}

func (k *FrNTTKernel) Close() {
	if k == nil {
		return
	}
	if k.vectorKernel != nil {
		k.vectorKernel.Close()
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

func (k *FrNTTKernel) RunStage(input, twiddles []curvegpu.U32x8, m int) ([]curvegpu.U32x8, error) {
	out, _, err := k.RunStageProfiled(input, twiddles, m)
	return out, err
}

func (k *FrNTTKernel) RunStageProfiled(input, twiddles []curvegpu.U32x8, m int) ([]curvegpu.U32x8, FrNTTStageProfile, error) {
	count := len(input)
	if count == 0 {
		return nil, FrNTTStageProfile{}, nil
	}
	if count&(count-1) != 0 {
		return nil, FrNTTStageProfile{}, fmt.Errorf("ntt stage requires power-of-two count, got %d", count)
	}
	if m <= 0 || count%(2*m) != 0 {
		return nil, FrNTTStageProfile{}, fmt.Errorf("invalid stage width m=%d for count=%d", m, count)
	}
	if len(twiddles) != m {
		return nil, FrNTTStageProfile{}, fmt.Errorf("twiddle length mismatch: got=%d want=%d", len(twiddles), m)
	}
	totalStart := time.Now()
	dataSize := uint64(count * frElementBytes)
	twiddleSize := uint64(len(twiddles) * frElementBytes)

	inputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-input",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuffer.Release()

	twiddleBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-twiddles",
		Size:  twiddleSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create twiddle buffer: %w", err)
	}
	defer twiddleBuffer.Release()

	outputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-output",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create output buffer: %w", err)
	}
	defer outputBuffer.Release()

	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-staging",
		Size:  dataSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-params",
		Size:  frNTTUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create uniform buffer: %w", err)
	}
	defer uniform.Release()

	uploadStart := time.Now()
	if err := k.device.Queue().WriteBuffer(inputBuffer, 0, flattenBatch(input)); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write input buffer: %w", err)
	}
	if err := k.device.Queue().WriteBuffer(twiddleBuffer, 0, flattenBatch(twiddles)); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write twiddle buffer: %w", err)
	}

	params := make([]byte, frNTTUniformBytes)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(m))
	if err := k.device.Queue().WriteBuffer(uniform, 0, params); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("write uniform buffer: %w", err)
	}

	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bn254-fr-ntt-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuffer, Size: dataSize},
			{Binding: 1, Buffer: twiddleBuffer, Size: twiddleSize},
			{Binding: 2, Buffer: outputBuffer, Size: dataSize},
			{Binding: 3, Buffer: uniform, Size: frNTTUniformBytes},
		},
	})
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create bind group: %w", err)
	}
	defer bindGroup.Release()
	uploadElapsed := time.Since(uploadStart)

	kernelStart := time.Now()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("create command encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32(((count/2)+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(outputBuffer, 0, staging, 0, dataSize)

	cmd, err := encoder.Finish()
	if err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("submit: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("wait idle: %w", err)
	}
	kernelElapsed := time.Since(kernelStart)

	readbackStart := time.Now()
	data := make([]byte, dataSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, FrNTTStageProfile{}, fmt.Errorf("read output buffer: %w", err)
	}
	readbackElapsed := time.Since(readbackStart)
	return unflattenBatch(data, count), FrNTTStageProfile{
		Upload:   uploadElapsed,
		Kernel:   kernelElapsed,
		Readback: readbackElapsed,
		Total:    time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) ForwardDIT(input []curvegpu.U32x8, stageTwiddles [][]curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	out, _, err := k.ForwardDITProfiled(input, stageTwiddles)
	return out, err
}

func (k *FrNTTKernel) ForwardDITProfiled(input []curvegpu.U32x8, stageTwiddles [][]curvegpu.U32x8) ([]curvegpu.U32x8, FrNTTProfile, error) {
	totalStart := time.Now()
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	state, bitReverse, err := k.vectorKernel.RunProfiled(FrVectorOpBitReverseCopy, input, nil, uint32(len(stageTwiddles)))
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("bit reverse input: %w", err)
	}
	var stages FrNTTStageProfile
	for _, twiddles := range stageTwiddles {
		var stageProfile FrNTTStageProfile
		state, stageProfile, err = k.RunStageProfiled(state, twiddles, len(twiddles))
		if err != nil {
			return nil, FrNTTProfile{}, err
		}
		stages.Upload += stageProfile.Upload
		stages.Kernel += stageProfile.Kernel
		stages.Readback += stageProfile.Readback
		stages.Total += stageProfile.Total
	}
	return state, FrNTTProfile{
		BitReverse: bitReverse,
		Stages:     stages,
		Total:      time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) InverseDIT(input []curvegpu.U32x8, inverseStageTwiddles [][]curvegpu.U32x8, scale curvegpu.U32x8) ([]curvegpu.U32x8, error) {
	out, _, err := k.InverseDITProfiled(input, inverseStageTwiddles, scale)
	return out, err
}

func (k *FrNTTKernel) InverseDITProfiled(input []curvegpu.U32x8, inverseStageTwiddles [][]curvegpu.U32x8, scale curvegpu.U32x8) ([]curvegpu.U32x8, FrNTTProfile, error) {
	totalStart := time.Now()
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	state, bitReverse, err := k.vectorKernel.RunProfiled(FrVectorOpBitReverseCopy, input, nil, uint32(len(inverseStageTwiddles)))
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("bit reverse inverse input: %w", err)
	}
	var stages FrNTTStageProfile
	for _, twiddles := range inverseStageTwiddles {
		var stageProfile FrNTTStageProfile
		state, stageProfile, err = k.RunStageProfiled(state, twiddles, len(twiddles))
		if err != nil {
			return nil, FrNTTProfile{}, err
		}
		stages.Upload += stageProfile.Upload
		stages.Kernel += stageProfile.Kernel
		stages.Readback += stageProfile.Readback
		stages.Total += stageProfile.Total
	}
	scaleBatch := make([]curvegpu.U32x8, len(state))
	for i := range scaleBatch {
		scaleBatch[i] = scale
	}
	scaled, scaleProfile, err := k.vectorKernel.RunProfiled(FrVectorOpMulFactors, state, scaleBatch, 0)
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("scale inverse output: %w", err)
	}
	return scaled, FrNTTProfile{
		BitReverse: bitReverse,
		Stages:     stages,
		Scale:      scaleProfile,
		Total:      time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) ForwardDITFullPathProfiled(input []curvegpu.U32x8, stageTwiddles [][]curvegpu.U32x8) ([]curvegpu.U32x8, FrNTTProfile, error) {
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	return k.fullPathTransform(input, stageTwiddles, nil, curvegpu.U32x8{}, false)
}

func (k *FrNTTKernel) InverseDITFullPathProfiled(input []curvegpu.U32x8, inverseStageTwiddles [][]curvegpu.U32x8, scale curvegpu.U32x8) ([]curvegpu.U32x8, FrNTTProfile, error) {
	if len(input) == 0 {
		return nil, FrNTTProfile{}, nil
	}
	return k.fullPathTransform(input, inverseStageTwiddles, nil, scale, true)
}

func (k *FrNTTKernel) fullPathTransform(input []curvegpu.U32x8, stageTwiddles [][]curvegpu.U32x8, aux []curvegpu.U32x8, scale curvegpu.U32x8, applyScale bool) ([]curvegpu.U32x8, FrNTTProfile, error) {
	count := len(input)
	totalStart := time.Now()
	dataSize := uint64(count * frElementBytes)
	logCount, err := bitReverseLogCount(count)
	if err != nil {
		return nil, FrNTTProfile{}, err
	}
	inputBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-input",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuffer.Release()
	stateA, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-state-a",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create stateA buffer: %w", err)
	}
	defer stateA.Release()
	stateB, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-state-b",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create stateB buffer: %w", err)
	}
	defer stateB.Release()
	auxBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-aux",
		Size:  dataSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create aux buffer: %w", err)
	}
	defer auxBuffer.Release()
	vectorUniform, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-vector-params",
		Size:  frVectorUniformBytes,
		Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create vector uniform buffer: %w", err)
	}
	defer vectorUniform.Release()
	staging, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "bn254-fr-ntt-full-staging",
		Size:  dataSize,
		Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead,
	})
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create staging buffer: %w", err)
	}
	defer staging.Release()

	uploadStart := time.Now()
	if err := k.device.Queue().WriteBuffer(inputBuffer, 0, flattenBatch(input)); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("write input buffer: %w", err)
	}
	auxBatch := normalizeBatch(aux, count)
	if err := k.device.Queue().WriteBuffer(auxBuffer, 0, flattenBatch(auxBatch)); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("write aux buffer: %w", err)
	}
	type stageBufferSet struct {
		twiddles *wgpu.Buffer
		uniform  *wgpu.Buffer
		size     uint64
	}
	stageBuffers := make([]stageBufferSet, len(stageTwiddles))
	defer func() {
		for _, stage := range stageBuffers {
			if stage.twiddles != nil {
				stage.twiddles.Release()
			}
			if stage.uniform != nil {
				stage.uniform.Release()
			}
		}
	}()
	for i, twiddles := range stageTwiddles {
		stageSize := uint64(len(twiddles) * frElementBytes)
		twiddleBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "bn254-fr-ntt-full-twiddles",
			Size:  stageSize,
			Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			return nil, FrNTTProfile{}, fmt.Errorf("create twiddle buffer: %w", err)
		}
		uniformBuffer, err := k.device.CreateBuffer(&wgpu.BufferDescriptor{
			Label: "bn254-fr-ntt-full-params",
			Size:  frNTTUniformBytes,
			Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst,
		})
		if err != nil {
			twiddleBuffer.Release()
			return nil, FrNTTProfile{}, fmt.Errorf("create ntt uniform buffer: %w", err)
		}
		if err := k.device.Queue().WriteBuffer(twiddleBuffer, 0, flattenBatch(twiddles)); err != nil {
			twiddleBuffer.Release()
			uniformBuffer.Release()
			return nil, FrNTTProfile{}, fmt.Errorf("write twiddle buffer: %w", err)
		}
		nttParams := make([]byte, frNTTUniformBytes)
		binary.LittleEndian.PutUint32(nttParams[0:], uint32(count))
		binary.LittleEndian.PutUint32(nttParams[4:], uint32(len(twiddles)))
		if err := k.device.Queue().WriteBuffer(uniformBuffer, 0, nttParams); err != nil {
			twiddleBuffer.Release()
			uniformBuffer.Release()
			return nil, FrNTTProfile{}, fmt.Errorf("write ntt params: %w", err)
		}
		stageBuffers[i] = stageBufferSet{
			twiddles: twiddleBuffer,
			uniform:  uniformBuffer,
			size:     stageSize,
		}
	}
	uploadElapsed := time.Since(uploadStart)

	kernelStart := time.Now()
	current := stateA
	next := stateB

	vectorParams := make([]byte, frVectorUniformBytes)
	binary.LittleEndian.PutUint32(vectorParams[0:], uint32(count))
	binary.LittleEndian.PutUint32(vectorParams[4:], uint32(FrVectorOpBitReverseCopy))
	binary.LittleEndian.PutUint32(vectorParams[8:], logCount)
	if err := k.device.Queue().WriteBuffer(vectorUniform, 0, vectorParams); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("write vector params: %w", err)
	}
	uploadElapsed += time.Since(uploadStart) - uploadElapsed
	if err := k.dispatchVectorBuffers(inputBuffer, auxBuffer, current, vectorUniform, dataSize); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("dispatch bit reverse: %w", err)
	}

	var stagesProfile FrNTTStageProfile
	for i := range stageTwiddles {
		stageKernelStart := time.Now()
		if err := k.dispatchNTTStageBuffers(current, stageBuffers[i].twiddles, next, stageBuffers[i].uniform, dataSize, stageBuffers[i].size, count); err != nil {
			return nil, FrNTTProfile{}, fmt.Errorf("dispatch stage: %w", err)
		}
		stagesProfile.Kernel += time.Since(stageKernelStart)
		current, next = next, current
	}

	var scaleProfile FrVectorProfile
	if applyScale {
		scaleBatch := make([]curvegpu.U32x8, count)
		for i := range scaleBatch {
			scaleBatch[i] = scale
		}
		scaleUploadStart := time.Now()
		if err := k.device.Queue().WriteBuffer(auxBuffer, 0, flattenBatch(scaleBatch)); err != nil {
			return nil, FrNTTProfile{}, fmt.Errorf("write scale batch: %w", err)
		}
		scaleProfile.Upload = time.Since(scaleUploadStart)
		vectorParams = make([]byte, frVectorUniformBytes)
		binary.LittleEndian.PutUint32(vectorParams[0:], uint32(count))
		binary.LittleEndian.PutUint32(vectorParams[4:], uint32(FrVectorOpMulFactors))
		scaleUploadStart = time.Now()
		if err := k.device.Queue().WriteBuffer(vectorUniform, 0, vectorParams); err != nil {
			return nil, FrNTTProfile{}, fmt.Errorf("write scale params: %w", err)
		}
		scaleProfile.Upload += time.Since(scaleUploadStart)
		scaleKernelStart := time.Now()
		if err := k.dispatchVectorBuffers(current, auxBuffer, next, vectorUniform, dataSize); err != nil {
			return nil, FrNTTProfile{}, fmt.Errorf("dispatch scale: %w", err)
		}
		scaleProfile.Kernel = time.Since(scaleKernelStart)
		current, next = next, current
		scaleProfile.Total = scaleProfile.Upload + scaleProfile.Kernel
	}

	if err := k.device.WaitIdle(); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("wait idle: %w", err)
	}
	kernelElapsed := time.Since(kernelStart)
	stagesProfile.Total = stagesProfile.Upload + stagesProfile.Kernel

	readbackStart := time.Now()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("create readback encoder: %w", err)
	}
	encoder.CopyBufferToBuffer(current, 0, staging, 0, dataSize)
	cmd, err := encoder.Finish()
	if err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("finish readback encoder: %w", err)
	}
	if _, err := k.device.Queue().Submit(cmd); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("submit readback: %w", err)
	}
	if err := k.device.WaitIdle(); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("wait readback idle: %w", err)
	}
	data := make([]byte, dataSize)
	if err := k.device.Queue().ReadBuffer(staging, 0, data); err != nil {
		return nil, FrNTTProfile{}, fmt.Errorf("read output buffer: %w", err)
	}
	readbackElapsed := time.Since(readbackStart)

	return unflattenBatch(data, count), FrNTTProfile{
		BitReverse: FrVectorProfile{
			Total: 0,
		},
		Stages: FrNTTStageProfile{
			Upload:   uploadElapsed + stagesProfile.Upload,
			Kernel:   kernelElapsed,
			Readback: readbackElapsed,
			Total:    time.Since(totalStart),
		},
		Scale: scaleProfile,
		Total: time.Since(totalStart),
	}, nil
}

func (k *FrNTTKernel) dispatchVectorBuffers(inputBuffer, auxBuffer, outputBuffer, uniformBuffer *wgpu.Buffer, dataSize uint64) error {
	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bn254-fr-ntt-full-vector-bg",
		Layout: k.vectorKernel.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuffer, Size: dataSize},
			{Binding: 1, Buffer: auxBuffer, Size: dataSize},
			{Binding: 2, Buffer: outputBuffer, Size: dataSize},
			{Binding: 3, Buffer: uniformBuffer, Size: frVectorUniformBytes},
		},
	})
	if err != nil {
		return err
	}
	defer bindGroup.Release()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(k.vectorKernel.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32((dataSize/uint64(frElementBytes)+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return err
	}
	cmd, err := encoder.Finish()
	if err != nil {
		return err
	}
	_, err = k.device.Queue().Submit(cmd)
	return err
}

func (k *FrNTTKernel) dispatchNTTStageBuffers(inputBuffer, twiddleBuffer, outputBuffer, uniformBuffer *wgpu.Buffer, dataSize, twiddleSize uint64, count int) error {
	bindGroup, err := k.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label:  "bn254-fr-ntt-full-bg",
		Layout: k.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: inputBuffer, Size: dataSize},
			{Binding: 1, Buffer: twiddleBuffer, Size: twiddleSize},
			{Binding: 2, Buffer: outputBuffer, Size: dataSize},
			{Binding: 3, Buffer: uniformBuffer, Size: frNTTUniformBytes},
		},
	})
	if err != nil {
		return err
	}
	defer bindGroup.Release()
	encoder, err := k.device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return err
	}
	pass.SetPipeline(k.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	pass.Dispatch(uint32(((count/2)+frWorkgroup-1)/frWorkgroup), 1, 1)
	if err := pass.End(); err != nil {
		return err
	}
	cmd, err := encoder.Finish()
	if err != nil {
		return err
	}
	_, err = k.device.Queue().Submit(cmd)
	return err
}
