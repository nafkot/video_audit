import sqlite3

def check_db():
    conn = sqlite3.connect('youtube_insights.db')
    c = conn.cursor()

    print("--- Database Diagnostics ---")

    # 1. Count Videos
    c.execute("SELECT count(*) FROM videos")
    video_count = c.fetchone()[0]
    print(f"Total Videos stored: {video_count}")

    # 2. Count Searchable Chunks (Segments)
    c.execute("SELECT count(*) FROM video_segments")
    segment_count = c.fetchone()[0]
    print(f"Total Transcript Segments: {segment_count}")

    if segment_count == 0 and video_count > 0:
        print("\n⚠️  CRITICAL ISSUE FOUND: You have videos, but NO transcripts.")
        print("    The Search will fail because there is no text to search through.")
        print("    This likely happened because the audio download failed during ingestion.")
    
    # 3. Check for specific keywords
    keyword = "immune"
    print(f"\n--- Searching for '{keyword}' directly in DB ---")
    
    # Check Segments
    c.execute("SELECT count(*) FROM video_segments WHERE text LIKE ?", (f'%{keyword}%',))
    seg_match = c.fetchone()[0]
    print(f"Matches in Transcripts: {seg_match}")

    # Check Summaries
    c.execute("SELECT title FROM videos WHERE overall_summary LIKE ?", (f'%{keyword}%',))
    summary_matches = c.fetchall()
    print(f"Matches in Summaries: {len(summary_matches)}")
    for row in summary_matches:
        print(f" - Found in summary of: {row[0]}")

    conn.close()

if __name__ == "__main__":
    check_db()
