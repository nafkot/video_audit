package analyze

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings" // Added strings package for helper functions
)

type VocalsAnalysis struct {
	Language         string                 `json:"language"`
	Transcript       string                 `json:"transcript"`
	Sentiment        map[string]interface{} `json:"sentiment"`
	Topic            map[string]interface{} `json:"topic"`
	OffensiveContent bool                   `json:"offensive_content"`
	OffensiveWords   []string               `json:"offensive_words"`
}

type AnalysisResult struct {
	Video              string          `json:"video"`
	Found              bool            `json:"found"`
	Similarity         int             `json:"similarity"`
	DetectedLowVolume  bool            `json:"detected_low_volume"`
	LowVolumePercent   int             `json:"low_volume_percent"`
	AIGenerated        bool            `json:"ai_generated"`
	AIConfidence       int             `json:"ai_confidence"`
	Folder             string          `json:"folder"`
	VocalsAnalysis     *VocalsAnalysis `json:"vocals_analysis,omitempty"`
	VideoDuration      float64         `json:"video_duration"`
	AudioTrackDuration float64         `json:"audio_track_duration"`
	DetectedDuration   float64         `json:"detected_duration"`
}

// Analyze runs the Python analyzer on a video audio WAV and target audio file
func Analyze(videoWavPath, targetAudioPath string, videoID string, robust bool, vocals bool, threshold int, skipAudioMatching bool) (AnalysisResult, error) {
	// Find python/analyze.py relative to the executable or working directory
	pythonScript := findPythonScript()
	if pythonScript == "" {
		return AnalysisResult{}, fmt.Errorf("python/analyze.py not found")
	}

	// Find Python interpreter (prefers venv)
	pythonExec := findPythonInterpreter()

	// 1. Create temporary output file path
	tempOutputFile, err := os.CreateTemp("", "audio_analysis_*.json")
	if err != nil {
		return AnalysisResult{}, fmt.Errorf("failed to create temporary output file: %w", err)
	}
	tempOutputFilePath := tempOutputFile.Name()
	tempOutputFile.Close() // Close file handle immediately

	// Ensure the temp file is removed after the function finishes
	defer os.Remove(tempOutputFilePath)

	// Build command arguments (args)
	args := []string{
		pythonScript,
		"--video-audio", videoWavPath,
		"--video-id", videoID,
	}

	// Only add target audio if not skipping audio matching
	if !skipAudioMatching && targetAudioPath != "" {
		args = append(args, "--target-audio", targetAudioPath)
		args = append(args, "--threshold", strconv.Itoa(threshold))
	}

	if robust {
		args = append(args, "--robust")
	}
	if vocals {
		args = append(args, "--vocals")
	}

	// 2. Add the required --output-file argument
	args = append(args, "--output-file", tempOutputFilePath)

	// --- VENV SHELL WRAPPER FIX START ---

	// Get the VENV root path (two directories up from the pythonExec path)
	venvDir := filepath.Dir(filepath.Dir(pythonExec))
	venvActivateScript := filepath.Join(venvDir, "bin", "activate")

	var cmd *exec.Cmd

	if _, err := os.Stat(venvActivateScript); err == nil {
		// Use the shell wrapper for reliable VENV execution (runs 'source activate' first)
		pythonCommandString := strings.Join(args, " ")

		// Full command: source /path/to/venv/bin/activate && /path/to/venv/bin/python3 analyze.py ...
		fullCommand := fmt.Sprintf("source %s && %s %s", venvActivateScript, pythonExec, pythonCommandString)

		// Execute via /bin/sh -c
		cmd = exec.Command("/bin/bash", "-c", fullCommand)
		fmt.Fprintln(os.Stderr, "[DEBUG] Using VENV shell wrapper for execution (via bash).")

	} else {
		// Fall back to direct execution if activate script is not found
		cmd = exec.Command(pythonExec, args...)
		fmt.Fprintln(os.Stderr, "[DEBUG] VENV activate script not found. Using direct execution.")
	}
	// --- VENV SHELL WRAPPER FIX END ---


	// 4. Run the command and check for errors
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Check if stderr has useful info
		if exitErr, ok := err.(*exec.ExitError); ok {
			// Read combined stdout/stderr output from the Python failure for detailed debugging
			pythonOutput := string(output)
			if pythonOutput != "" {
				// Return the full Python error message
				return AnalysisResult{}, fmt.Errorf("python analyzer failed: %s", pythonOutput)
			}
			// Fallback if output is empty but process failed
			return AnalysisResult{}, fmt.Errorf("python analyzer failed with exit code %d (empty output)", exitErr.ExitCode())
		}
		return AnalysisResult{}, fmt.Errorf("python analyzer failed: %w", err)
	}

	// 5. Read JSON result from the file
	data, err := os.ReadFile(tempOutputFilePath)
	if err != nil {
		return AnalysisResult{}, fmt.Errorf("failed to read analyzer output file: %w", err)
	}

	// 6. Parse JSON result
	var result AnalysisResult
	if err := json.Unmarshal(data, &result); err != nil {
		// Include the file content in the error for debugging if parsing fails
		return AnalysisResult{}, fmt.Errorf("failed to parse analyzer output: %w. Raw content: %s", err, string(data))
	}

	return result, nil
}

// findPythonInterpreter finds the best Python interpreter to use (prefers venv)
func findPythonInterpreter() string {
	// Try VENV locations specific to the working directory structure
	venvCandidates := []string{
		// 1. **Highest Priority:** Check for the active 'venv_detection' folder
		"venv_detection/bin/python3",
		"venv_detection/bin/python",

		// 2. Standard internal venv paths (low priority, just in case)
		"python/.venv/bin/python3",
		"../python/.venv/bin/python3",
	}

	// Try relative to executable
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		venvCandidates = append(venvCandidates,
			filepath.Join(exeDir, "venv_detection", "bin", "python3"), // Target specific VENV
			filepath.Join(exeDir, "..", "venv_detection", "bin", "python3"),
		)
	}

	for _, path := range venvCandidates {
		if _, err := os.Stat(path); err == nil {
			absPath, _ := filepath.Abs(path)
			// Print debug info to stderr if successful
			// Note: We avoid printing Fprintf here to prevent potential race conditions with Go's own Fprintf
			return absPath
		}
	}

	// Fall back to system python3
	return "python3"
}

// findPythonScript searches for python/analyze.py in common locations
func findPythonScript() string {
	// Try relative to working directory
	candidates := []string{
		"python/analyze.py",
		"../python/analyze.py",
		"../../python/analyze.py",
	}

	// Try relative to executable
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		candidates = append(candidates, filepath.Join(exeDir, "python", "analyze.py"))
		candidates = append(candidates, filepath.Join(exeDir, "..", "python", "analyze.py"))
	}

	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	return ""
}
