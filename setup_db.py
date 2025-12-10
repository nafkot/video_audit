import sqlite3

def init_db():
    conn = sqlite3.connect('youtube_insights.db')
    c = conn.cursor()

    # 1. Channels Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            subscriber_count INTEGER,
            video_count INTEGER,
            view_count INTEGER,
            creation_date TEXT,
            category TEXT,
            thumbnail_url TEXT
        )
    ''')

    # 2. Videos Table (Expanded with new metadata)
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            title TEXT,
            channel_name TEXT,
            upload_date TEXT,
            duration INTEGER,
            overall_summary TEXT,
            overall_sentiment TEXT,
            topics TEXT,
            brands TEXT,
            sponsors TEXT,

            -- NEW COLUMNS
            view_count INTEGER,
            like_count INTEGER,
            thumbnail_url TEXT,
            author TEXT,
            is_family_safe BOOLEAN,
            owner_profile_url TEXT,
            category TEXT,

            FOREIGN KEY(channel_id) REFERENCES channels(channel_id)
        )
    ''')

    # 3. Segments Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS video_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            start_time REAL,
            end_time REAL,
            text TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(video_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database updated with new columns.")

if __name__ == "__main__":
    init_db()
