import sqlite3

def check_channels():
    conn = sqlite3.connect('youtube_insights.db')
    c = conn.cursor()
    
    print("--- Checking Channels Table ---")
    try:
        c.execute("SELECT * FROM channels")
        channels = c.fetchall()
        print(f"Total Channels found: {len(channels)}")
        for ch in channels:
            # Adjust indices based on your table schema
            # Assuming: channel_id, title, ...
            print(f" - ID: {ch[0]}, Title: {ch[1]}") 
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")
        print("The 'channels' table might not exist. Did you run 'setup_db.py'?")

    conn.close()

if __name__ == "__main__":
    check_channels()
