package curvegpu

import "fmt"

func SplitWords4(words [4]uint64) U32x8 {
	var out U32x8
	for i, word := range words {
		out[2*i] = uint32(word)
		out[2*i+1] = uint32(word >> 32)
	}
	return out
}

func JoinWords8(limbs U32x8) [4]uint64 {
	var out [4]uint64
	for i := range out {
		lo := uint64(limbs[2*i])
		hi := uint64(limbs[2*i+1]) << 32
		out[i] = lo | hi
	}
	return out
}

func SplitWords6(words [6]uint64) U32x12 {
	var out U32x12
	for i, word := range words {
		out[2*i] = uint32(word)
		out[2*i+1] = uint32(word >> 32)
	}
	return out
}

func JoinWords12(limbs U32x12) [6]uint64 {
	var out [6]uint64
	for i := range out {
		lo := uint64(limbs[2*i])
		hi := uint64(limbs[2*i+1]) << 32
		out[i] = lo | hi
	}
	return out
}

func SplitUint64Words(words []uint64) ([]uint32, error) {
	switch len(words) {
	case 4:
		var fixed [4]uint64
		copy(fixed[:], words)
		out := SplitWords4(fixed)
		return out[:], nil
	case 6:
		var fixed [6]uint64
		copy(fixed[:], words)
		out := SplitWords6(fixed)
		return out[:], nil
	default:
		return nil, fmt.Errorf("unsupported host word count %d", len(words))
	}
}

func JoinUint32Limbs(limbs []uint32) ([]uint64, error) {
	switch len(limbs) {
	case 8:
		var fixed U32x8
		copy(fixed[:], limbs)
		out := JoinWords8(fixed)
		return out[:], nil
	case 12:
		var fixed U32x12
		copy(fixed[:], limbs)
		out := JoinWords12(fixed)
		return out[:], nil
	default:
		return nil, fmt.Errorf("unsupported gpu limb count %d", len(limbs))
	}
}
