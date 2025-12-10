package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"path/filepath"

	"syncmusic/internal/analyze"
	"syncmusic/internal/media"

	"github.com/spf13/cobra"
)

var (
	videoFiles  string
	audioFile   string
	threads     int
	robust      bool
	vocals      bool
	threshold   int
	verbose     bool
	validation  bool
)

type VideoResult struct {
	Video              string                  `json:"video"`
	Found              *bool                   `json:"found,omitempty"`
	Similarity         *int                    `json:"similarity,omitempty"`
	Volume             *int                    `json:"volume,omitempty"`
	AIGenerated        bool                    `json:"ai_generated"`
	AIConfidence       int                     `json:"ai_confidence"`
	Folder             string                  `json:"folder,omitempty"`
	VocalsAnalysis     *analyze.VocalsAnalysis `json:"vocals_analysis,omitempty"`
	VideoDuration      float64                 `json:"video_duration"`
	AudioTrackDuration *float64                `json:"audio_track_duration,omitempty"`
	DetectedDuration   *float64                `json:"detected_duration,omitempty"`
	Error              string                  `json:"error,omitempty"`
}

var rootCmd = &cobra.Command{
	Use:   "syncmusic",
	Short: "Detect music in short videos with voice-over resistance",
	Long:  `syncmusic analyzes short videos to detect if a target audio track appears, even with loud voice-overs.`,
	RunE:  run,
}

// Execute executes the root command
func Execute() error {
	return rootCmd.Execute()
}

// init initializes the root command
func init() {
	rootCmd.Flags().StringVar(&videoFiles, "video", "", "comma-separated list of MP4 video paths (required)")
	rootCmd.Flags().StringVar(&audioFile, "audio", "", "path to target audio file (optional - enables audio matching analysis)")
	rootCmd.Flags().IntVar(&threads, "threads", runtime.NumCPU(), "number of concurrent threads")
	rootCmd.Flags().BoolVar(&robust, "robust", true, "enable Demucs + DTW stage for voice-over resistance (slower)")
	rootCmd.Flags().BoolVar(&vocals, "vocals", false, "target audio has vocals/singer (not instrumental)")
	rootCmd.Flags().IntVar(&threshold, "threshold", 70, "similarity threshold (0-100) to mark found as true")
	rootCmd.Flags().BoolVar(&verbose, "verbose", false, "enable verbose error logging")
	rootCmd.Flags().BoolVar(&validation, "validation", false, "enable video duration validation (15-70 seconds)")

	rootCmd.MarkFlagRequired("video")
}

// run runs the root command
func run(cmd *cobra.Command, args []string) error {
	// Check if audio file is provided and exists
	skipAudioAnalysis := audioFile == ""
	if !skipAudioAnalysis {
		if _, err := os.Stat(audioFile); os.IsNotExist(err) {
			return fmt.Errorf("audio file not found: %s", audioFile)
		}
	}

	videoList := strings.Split(videoFiles, ",")

	if len(videoList) == 0 {
		return fmt.Errorf("no video files provided")
	}

	for i := range videoList {
		videoList[i] = strings.TrimSpace(videoList[i])
	}

	results := make([]VideoResult, len(videoList))
	var wg sync.WaitGroup

	semaphore := make(chan struct{}, threads)

	for i, videoPath := range videoList {
		wg.Add(1)

		go func(idx int, path string) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			results[idx] = processVideo(path, audioFile, robust, vocals, threshold, skipAudioAnalysis, validation)
		}(i, videoPath)
	}
	wg.Wait()

	output, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal results: %w", err)
	}

	fmt.Println(string(output))
	return nil
}

func processVideo(videoPath, audioPath string, robust bool, vocals bool, threshold int, skipAudioAnalysis bool, validateDuration bool) VideoResult {
	// Validate video file exists
	if err := validateVideo(videoPath); err != nil {
		return newErrorResult(videoPath, err)
	}

	// Extract the video ID from the path (e.g., JQF7GzmtSEY from storage/videos/JQF7GzmtSEY.mp4)
	videoID := filepath.Base(videoPath)
	videoID = strings.TrimSuffix(videoID, filepath.Ext(videoID)) // Remove extension (.mp4)

	duration, err := media.GetDuration(videoPath)
	if err != nil {
		return newErrorResult(videoPath, fmt.Errorf("failed to get video duration: %w", err))
	}

	// Validate duration range only if validation flag is enabled
	if validateDuration && (duration < 15 || duration > 70) {
		return newErrorResult(videoPath, fmt.Errorf("video duration %.1fs out of range (must be 15-70s)", duration))
	}

	// Extract audio from video
	tempWav, cleanup, err := media.ExtractAudio(videoPath)
	if err != nil {
		return newErrorResult(videoPath, fmt.Errorf("failed to extract audio: %w", err))
	}
	defer cleanup()

	// Run Python analyzer
    // FIX: Pass the newly defined videoID as the third argument.
	result, err := analyze.Analyze(tempWav, audioPath, videoID, robust, vocals, threshold, skipAudioAnalysis)
	if err != nil {
		return newErrorResult(videoPath, fmt.Errorf("analysis failed: %w", err))
	}

	// Log success
	logSuccess(videoPath, result, skipAudioAnalysis)

	// Build and return result
	return buildVideoResult(videoPath, result, skipAudioAnalysis)
}

// validateVideo checks if video file exists
func validateVideo(videoPath string) error {
	if _, err := os.Stat(videoPath); os.IsNotExist(err) {
		return fmt.Errorf("video file not found")
	}
	return nil
}

// newErrorResult creates a VideoResult with an error
func newErrorResult(videoPath string, err error) VideoResult {
	logError(videoPath, err.Error())
	return VideoResult{
		Video: videoPath,
		Error: err.Error(),
	}
}

// logError logs an error message if verbose mode is enabled
func logError(videoPath, message string) {
	if verbose {
		fmt.Fprintf(os.Stderr, "[%s] %s\n", videoPath, message)
	}
}

// logSuccess logs a success message if verbose mode is enabled
func logSuccess(videoPath string, result analyze.AnalysisResult, skipAudioAnalysis bool) {
	if !verbose {
		return
	}

	if skipAudioAnalysis {
		fmt.Fprintf(os.Stderr, "[%s] Success: no audio matching performed\n", videoPath)
	} else {
		fmt.Fprintf(os.Stderr, "[%s] Success: similarity=%d, found=%v\n", videoPath, result.Similarity, result.Found)
	}
}

// buildVideoResult constructs the VideoResult from analysis results
func buildVideoResult(videoPath string, result analyze.AnalysisResult, skipAudioAnalysis bool) VideoResult {
	videoResult := VideoResult{
		Video:          videoPath,
		AIGenerated:    result.AIGenerated,
		AIConfidence:   result.AIConfidence,
		Folder:         result.Folder,
		VocalsAnalysis: result.VocalsAnalysis,
		VideoDuration:  result.VideoDuration,
	}

	// Add audio-matching fields only if audio analysis was performed
	if !skipAudioAnalysis {
		videoResult.Found = &result.Found
		videoResult.Similarity = &result.Similarity
		videoResult.AudioTrackDuration = &result.AudioTrackDuration
		videoResult.DetectedDuration = &result.DetectedDuration

		if result.Found {
			volume := 100 - result.LowVolumePercent
			videoResult.Volume = &volume
		}
	}

	return videoResult
}
