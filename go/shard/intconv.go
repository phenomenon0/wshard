package shard

import "strconv"

const maxShardInt64 = 1<<63 - 1

func uint64ToInt(v uint64) (int, bool) {
	if strconv.IntSize == 32 {
		if v > 1<<31-1 {
			return 0, false
		}
	} else if v > maxShardInt64 {
		return 0, false
	}

	// #nosec G115 -- bounds checked above for the current platform int size.
	return int(v), true
}

func uint64ToInt64(v uint64) (int64, bool) {
	if v > maxShardInt64 {
		return 0, false
	}

	// #nosec G115 -- bounds checked above.
	return int64(v), true
}
