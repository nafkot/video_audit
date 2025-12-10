from flask import Flask, render_template, request, g
import sqlite3
import os
import json
import markdown
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

app = Flask(__name__)
DB_NAME = 'youtube_insights.db'
client = OpenAI()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_NAME)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Custom Filter to parse JSON strings in templates
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except:
        return []

def extract_keywords(user_query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Extract 1-3 search keywords."}, {"role": "user", "content": user_query}]
        )
        return response.choices[0].message.content.strip()
    except: return user_query 

def generate_answer(user_query, segments, is_comparison=False):
    if not segments: return None

    # 1. Build Context with Attribution
    context_text = ""
    unique_channels = set()

    # Increase limit for better comparison context
    for seg in segments[:35]:
        channel = seg['channel_name']
        time = int(seg['start_time'])
        text = seg['text']

        # Structure the evidence block clearly for the LLM
        context_text += f"SOURCE: {channel} (Time: {time}s)\nQUOTE: {text}\n\n"
        unique_channels.add(channel)

    channel_list_str = ", ".join(sorted(unique_channels))

    # 2. Dynamic System Prompt
    system_prompt = "You are a video analyst."

    if is_comparison:
        system_prompt += f" You are comparing these specific channels: {channel_list_str}."
        system_prompt += " \n\nCRITICAL RULES:"
        system_prompt += " 1. REFER TO CHANNELS BY NAME ONLY. Never say 'one channel' or 'the other'."
        system_prompt += " 2. If the evidence for a channel is weak, say 'The transcript for [Channel Name] does not mention this directly'."
        system_prompt += " 3. Structure your response with these headers:\n"
        system_prompt += "    **Comparison:** (Where they align)\n"
        system_prompt += "    **Contrast:** (Where they disagree or differ in focus)\n"
        system_prompt += " 4. Cite specific timestamps (e.g., '(24s)') when making a claim."
        system_prompt += " 5. When discussing specific content, like Unbox Therapy's coverage of audio devices or sleep tech, clearly state the channel name." # Added rule
    else:
        system_prompt += f" Answer the question based on the content from {channel_list_str}."
        system_prompt += " Use bold text for key insights."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User Question: {user_query}\n\nEvidence:\n{context_text}"}
            ]
        )
        raw_text = response.choices[0].message.content
        return markdown.markdown(raw_text)
    except Exception as e:
        return None

def get_active_channels(channel_ids_str):
    if not channel_ids_str: return []
    ids = channel_ids_str.split(',')
    conn = get_db()
    placeholders = ','.join('?' for _ in ids)
    cur = conn.execute(f"SELECT * FROM channels WHERE channel_id IN ({placeholders})", ids)
    return [dict(row) for row in cur.fetchall()]

def search_available_channels(query):
    conn = get_db()
    cur = conn.execute("SELECT * FROM channels WHERE title LIKE ? OR channel_id LIKE ?", (f'%{query}%', f'%{query}%'))
    return [dict(row) for row in cur.fetchall()]

@app.route('/')
def index():
    channel_ids_str = request.args.get('channels', '')
    topic_query = request.args.get('q', '')
    add_channel_query = request.args.get('add_channel_q', '')
    
    if add_channel_query:
        found_channels = search_available_channels(add_channel_query)
        return render_template('index.html', found_channels=found_channels, active_channels=get_active_channels(channel_ids_str), current_channel_ids=channel_ids_str)

    conn = get_db()
    active_channels = get_active_channels(channel_ids_str)
    active_ids = [c['channel_id'] for c in active_channels]
    
    videos = []
    segments = []
    ai_answer = None
    
    video_sql = "SELECT * FROM videos"
    video_params = []
    
    seg_sql = "SELECT v.title, v.channel_name, s.video_id, s.start_time, s.text FROM video_segments s JOIN videos v ON s.video_id = v.video_id"
    seg_params = []
    
    if active_ids:
        placeholders = ','.join('?' for _ in active_ids)
        video_sql += f" WHERE channel_id IN ({placeholders})"
        video_params.extend(active_ids)
        seg_sql += f" WHERE v.channel_id IN ({placeholders})"
        seg_params.extend(active_ids)
    
    if topic_query:
        if len(topic_query.split()) > 2: keywords = extract_keywords(topic_query)
        else: keywords = topic_query
        search_terms = keywords.split()
        
        prefix = " AND" if "WHERE" in video_sql else " WHERE"
        video_sql += prefix + " (" + " OR ".join(["title LIKE ? OR overall_summary LIKE ? OR topics LIKE ?" for _ in search_terms]) + ")"
        for term in search_terms: video_params.extend([f'%{term}%', f'%{term}%', f'%{term}%'])
            
        prefix_seg = " AND" if "WHERE" in seg_sql else " WHERE"
        seg_sql += prefix_seg + " (" + " OR ".join(["s.text LIKE ?" for _ in search_terms]) + ")"
        for term in search_terms: seg_params.append(f'%{term}%')
        seg_sql += " LIMIT 30"
    
    if topic_query or active_ids:
        cur = conn.execute(video_sql, video_params)
        videos = cur.fetchall()
        if topic_query:
            cur_seg = conn.execute(seg_sql, seg_params)
            segments = cur_seg.fetchall()
            if segments:
                is_comparison = len(active_ids) > 1
                ai_answer = generate_answer(topic_query, segments, is_comparison)

    return render_template('index.html', videos=videos, segments=segments, query=topic_query, ai_answer=ai_answer, active_channels=active_channels, current_channel_ids=channel_ids_str)

@app.route('/channel/<channel_id>')
def channel_profile(channel_id):
    conn = get_db()
    cur = conn.execute("SELECT * FROM channels WHERE channel_id = ?", (channel_id,))
    channel = cur.fetchone()
    if not channel: return "Channel Not Found", 404
    
    # Get Videos
    cur_vids = conn.execute("SELECT * FROM videos WHERE channel_id = ? ORDER BY upload_date DESC", (channel_id,))
    videos = cur_vids.fetchall()
    
    # Aggregations
    topics = []
    brands = []
    sponsors = []
    sentiment_scores = []
    
    for v in videos:
        if v['topics']: topics.extend([t.strip() for t in v['topics'].split(',')])

        # Safe JSON parsing for brands
        if v['brands']:
            parsed_brands = json.loads(v['brands'])
            if parsed_brands: # Ensure it's not None
                brands.extend(parsed_brands)

        # Safe JSON parsing for sponsors
        if v['sponsors']:
            parsed_sponsors = json.loads(v['sponsors'])
            if parsed_sponsors: # Ensure it's not None
                sponsors.extend(parsed_sponsors)
        # Simple sentiment mapping
        if 'Positive' in v['overall_sentiment']: sentiment_scores.append(100)
        elif 'Negative' in v['overall_sentiment']: sentiment_scores.append(0)
        else: sentiment_scores.append(50)

    stats = {
        'top_topics': [t[0] for t in Counter(topics).most_common(10)],
        'top_brands': [b[0] for b in Counter(brands).most_common(20)],
        'sponsors': [s[0] for s in Counter(sponsors).most_common(5)],
        'sentiment_avg': sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 50
    }

    return render_template('profile.html', channel=channel, videos=videos, stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
