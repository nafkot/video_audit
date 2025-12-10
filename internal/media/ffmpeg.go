package media

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// ExtractAudio extracts audio from a video file to a temporary WAV file (44.1kHz, mono)
// Returns the path to the temp WAV file, a cleanup function, and any error
func ExtractAudio(videoPath string) (string, func(), error) {
	// Create temporary directory for this video
	tempDir, err := os.MkdirTemp("", "syncmusic-*")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create temp dir: %w", err)
	}

	cleanup := func() {
		os.RemoveAll(tempDir)
	}

	// Generate temp WAV path
	tempWav := filepath.Join(tempDir, "audio.wav")

	// Extract audio using ffmpeg
	cmd := exec.Command("ffmpeg",
		"-i", videoPath,
		"-vn", // No video
		"-acodec", "pcm_s16le", // PCM 16-bit
		"-ar", "44100", // 44.1kHz sample rate
		"-ac", "1", // Mono
		"-y", // Overwrite output
		tempWav,
	)

	// Suppress ffmpeg output unless there's an error
	if err := cmd.Run(); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("ffmpeg extraction failed: %w", err)
	}

	return tempWav, cleanup, nil
}
