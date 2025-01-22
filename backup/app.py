import streamlit as st
import pandas as pd
import sqlite3
from openai import OpenAI
import time
from datetime import datetime
import threading
from functools import wraps
import plotly.express as px

# Initialize OpenAI
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Constants
SHEET_COLUMNS = ["Tool Name", "URL", "GitHub URL", "Domain", "Pricing"]
DOMAIN_CATEGORIES = [
    "Ideas", "Presentation", "Website", "Writing", "AI Model",
    "Meeting", "Chatbot", "Automation", "UI/UX", "Image",
    "Design", "Video", "Blog Writer", "Marketing", "Twitter",
    "Code Generator", "Speech to Text", "AI Detector", "Voice", "Web3"
]

class TimeoutException(Exception):
    pass

def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutException("Timeout")]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutException("Function call timed out")
            
            if isinstance(result[0], Exception):
                raise result[0]
            
            return result[0]
        return wrapper
    return decorator

def init_db():
    """Initialize SQLite database and create table if it doesn't exist"""
    conn = sqlite3.connect('ai_tools.db')
    c = conn.cursor()
    
    # Main tools table
    c.execute('''CREATE TABLE IF NOT EXISTS ai_tools
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tool_name TEXT UNIQUE,
                  url TEXT,
                  github_url TEXT,
                  pricing TEXT,
                  added_date TEXT)''')
    
    # Domains table
    c.execute('''CREATE TABLE IF NOT EXISTS domains
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE)''')
    
    # Tool-Domain relationship table
    c.execute('''CREATE TABLE IF NOT EXISTS tool_domains
                 (tool_id INTEGER,
                  domain_id INTEGER,
                  FOREIGN KEY (tool_id) REFERENCES ai_tools(id),
                  FOREIGN KEY (domain_id) REFERENCES domains(id),
                  PRIMARY KEY (tool_id, domain_id))''')
    
    conn.commit()
    conn.close()

def migrate_db():
    """Migrate database to add ID column if it doesn't exist"""
    conn = sqlite3.connect('ai_tools.db')
    c = conn.cursor()
    try:
        # Check if we need to migrate
        c.execute("SELECT id FROM ai_tools LIMIT 1")
    except sqlite3.OperationalError:
        # Need to migrate - create temporary table
        c.execute("""CREATE TABLE ai_tools_temp
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     tool_name TEXT UNIQUE,
                     url TEXT,
                     github_url TEXT,
                     domain TEXT,
                     pricing TEXT,
                     added_date TEXT)""")
        
        # Copy data from old table to new table
        c.execute("""INSERT INTO ai_tools_temp 
                    (tool_name, url, github_url, domain, pricing, added_date)
                    SELECT tool_name, url, github_url, domain, pricing, added_date 
                    FROM ai_tools""")
        
        # Drop old table
        c.execute("DROP TABLE ai_tools")
        
        # Rename new table to original name
        c.execute("ALTER TABLE ai_tools_temp RENAME TO ai_tools")
        
        conn.commit()
    finally:
        conn.close()

def get_tools_data():
    """Fetch all tools from the database with their domains"""
    try:
        conn = sqlite3.connect('ai_tools.db')
        query = """
            SELECT 
                t.id,
                t.tool_name as 'Tool Name',
                t.url as 'URL',
                t.github_url as 'GitHub URL',
                GROUP_CONCAT(d.name) as 'Domain',
                t.pricing as 'Pricing'
            FROM ai_tools t
            LEFT JOIN tool_domains td ON t.id = td.tool_id
            LEFT JOIN domains d ON td.domain_id = d.id
            GROUP BY t.id
            ORDER BY t.id
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error fetching data from database: {str(e)}")
        return pd.DataFrame(columns=['id'] + SHEET_COLUMNS)

def add_tool(tool_data):
    """Add a new tool to the database with multiple domains"""
    try:
        conn = sqlite3.connect('ai_tools.db')
        c = conn.cursor()
        
        # Insert tool
        c.execute('''INSERT OR REPLACE INTO ai_tools 
                     (tool_name, url, github_url, pricing, added_date)
                     VALUES (?, ?, ?, ?, ?)''',
                 (tool_data[0], tool_data[1], tool_data[2], tool_data[4], datetime.now().isoformat()))
        
        tool_id = c.lastrowid if c.lastrowid else c.execute("SELECT id FROM ai_tools WHERE tool_name=?", (tool_data[0],)).fetchone()[0]
        
        # Handle domains (tool_data[3] should now be a list of domains)
        for domain in tool_data[3]:
            # Insert domain if not exists
            c.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,))
            domain_id = c.execute("SELECT id FROM domains WHERE name=?", (domain,)).fetchone()[0]
            
            # Link tool to domain
            c.execute("INSERT OR IGNORE INTO tool_domains (tool_id, domain_id) VALUES (?, ?)", (tool_id, domain_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error adding tool to database: {str(e)}")
        return False

def search_for_new_tools(existing_tools):
    prompt = f"""You are an AI assistant tasked with finding new AI tools. Please search for AI tools not in the following list:
    {existing_tools}
    
    Return exactly 10 new tools. For each tool, provide the following information in a structured format:
    1. Tool Name
    2. URL (must be a valid URL, not N/A)
    3. GitHub URL (if available, otherwise write 'N/A')
    4. Domain (one of: {', '.join(DOMAIN_CATEGORIES)})
    5. Pricing (Free, Freemium, Paid, or Enterprise)
    
    Return the information in CSV format, one tool per line.
    Only include verified, legitimate AI tools with working URLs.
    If you cannot find any new legitimate tools, respond with "No new tools found."
    Important: Only return tools you are confident exist and have working URLs."""

    @timeout_decorator(60)
    def make_api_call():
        return client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant skilled at researching AI tools. Only return verified tools with working URLs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

    try:
        response = make_api_call()
        content = response.choices[0].message.content
        
        # Check if no tools were found
        if "No new tools found" in content:
            return None
            
        # Validate the response
        new_tools_list = [line.split(',') for line in content.split('\n') if line.strip()]
        validated_tools = []
        
        for tool in new_tools_list:
            # Skip invalid entries
            if len(tool) != 5:
                continue
                
            tool_name, url, github_url, domain, pricing = [t.strip() for t in tool]
            
            # Validate URL (basic check)
            if url.lower() == 'n/a' or not url.startswith(('http://', 'https://')):
                continue
                
            # Validate domain
            if domain not in DOMAIN_CATEGORIES:
                continue
                
            # Validate pricing
            if pricing not in ['Free', 'Freemium', 'Paid', 'Enterprise']:
                continue
                
            validated_tools.append(tool)
        
        if not validated_tools:
            return None
            
        return '\n'.join(','.join(tool) for tool in validated_tools)
        
    except TimeoutException:
        st.warning("Search timed out after 60 seconds.")
        return None
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def recommend_tools(query, tools_df):
    # First check if we have any tools in the filtered dataset
    if tools_df.empty:
        return "Sorry, I couldn't find any tools in our database matching your requirements."

    # First, determine the most relevant domain based on the query
    domain_prompt = f"""Given this user need: "{query}"
    Which of these domains would be most relevant (pick up to 2):
    {', '.join(DOMAIN_CATEGORIES)}
    
    Return only the domain names, separated by comma if multiple."""

    try:
        domain_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a domain classifier."},
                     {"role": "user", "content": domain_prompt}],
            temperature=0.3
        )
        suggested_domains = [d.strip() for d in domain_response.choices[0].message.content.split(',')]
        
        # Filter tools by suggested domains - handle multiple domains per tool
        domain_filtered_df = tools_df[
            tools_df['Domain'].str.split(',').apply(
                lambda domains: any(
                    domain.strip().lower() in [sd.lower() for sd in suggested_domains]
                    for domain in domains
                )
            )
        ]
        
        # If no tools found in suggested domains, use all tools
        if domain_filtered_df.empty:
            domain_filtered_df = tools_df

    except Exception as e:
        domain_filtered_df = tools_df  # Fall back to all tools if domain classification fails

    prompt = f"""Given the following user need:
    "{query}"
    
    And this list of available AI tools:
    {domain_filtered_df.to_string()}
    
    Please recommend the most suitable tools from ONLY the above list. For each recommendation:
    1. Name the tool and include its URL in parentheses
    2. Explain why it's suitable for the user's needs
    3. List key features
    4. Mention pricing model
    
    Format each tool recommendation as:
    
    1. [Tool Name] (URL)
    - Why it's suitable: ...
    - Key features: ...
    - Pricing: ...
    
    If none of the tools match the user's needs, respond with: "Sorry, I couldn't find a suitable tool for your specific needs in our current database."
    
    Keep the response concise and practical."""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant that ONLY recommends tools from the provided database. Never suggest tools that are not in the given list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        recommendations = response.choices[0].message.content
        
        # Extract tool names for filtering the table
        recommended_tools = []
        lines = recommendations.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Format: "[Tool Name]"
            if "[" in line and "]" in line:
                tool_name = line[line.find("[")+1:line.find("]")]
                if tool_name in domain_filtered_df['Tool Name'].values:
                    recommended_tools.append(tool_name)
            
            # Format: "Tool: ToolName"
            elif "Tool:" in line:
                tool_name = line.split("Tool:")[1].strip()
                if tool_name in domain_filtered_df['Tool Name'].values:
                    recommended_tools.append(tool_name)
            
            # Format: "1. ToolName" or numbered list
            elif line and line[0].isdigit() and ". " in line:
                # Try to extract name before any description
                possible_name = line.split(". ")[1].split(":")[0].strip()
                if possible_name in domain_filtered_df['Tool Name'].values:
                    recommended_tools.append(possible_name)
            
            # Direct match with tool name
            else:
                for tool_name in domain_filtered_df['Tool Name'].values:
                    if tool_name in line:
                        recommended_tools.append(tool_name)
                        break
        
        # Update session state with recommended tools
        if recommended_tools:
            # Remove duplicates while preserving order
            recommended_tools = list(dict.fromkeys(recommended_tools))
            st.session_state.filtered_tools = domain_filtered_df[domain_filtered_df['Tool Name'].isin(recommended_tools)]
            st.session_state.suggested_domains = suggested_domains
            
            # Debug info (temporary)
            print("Found tools:", recommended_tools)
        else:
            # If no tools were found in the parsing, show all filtered data
            st.session_state.filtered_tools = domain_filtered_df
        
        return recommendations

    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def get_csv_download_link(df):
    """Generate a CSV file from the dataframe and create a download link"""
    csv = df.to_csv(index=False)
    return csv

def populate_initial_tools():
    tools = [ 
    # Ideas
    ["Bing Chat", "https://bing.com/chat", "N/A", ["Ideas"], "Free"],
    ["YOU", "https://you.com", "N/A", ["Ideas"], "Free"],
    ["Perplexity", "https://perplexity.ai", "N/A", ["Ideas"], "Freemium"],
    ["ChatGPT", "https://chat.openai.com", "N/A", ["Ideas", "Chatbot"], "Freemium"],
    ["Claude", "https://claude.ai", "N/A", ["Ideas", "Chatbot"], "Free"],
    
    # Presentation
    ["Prezi", "https://prezi.com", "N/A", ["Presentation"], "Freemium"],
    ["Pitch", "https://pitch.com", "N/A", ["Presentation"], "Freemium"],
    ["Popai.pro", "https://popai.pro", "N/A", ["Presentation"], "Paid"],
    ["Slides AI", "https://slidesai.io", "N/A", ["Presentation"], "Paid"],
    ["Slidebean", "https://slidebean.com", "N/A", ["Presentation"], "Paid"],
    
    # Website
    ["Dora", "N/A", "N/A", ["Website"], "N/A"],
    ["Durable", "https://durable.co", "N/A", ["Website"], "Freemium"],
    ["Wegic", "https://wegic.com", "N/A", ["Website"], "N/A"],
    ["Framer", "https://framer.com", "N/A", ["Website"], "Freemium"],
    ["10Web", "https://10web.io", "N/A", ["Website"], "Paid"],
    
    # Writing
    ["Rytr", "https://rytr.me", "N/A", ["Writing"], "Freemium"],
    ["Jasper", "https://jasper.ai", "N/A", ["Writing"], "Paid"],
    ["Copy AI", "https://copy.ai", "N/A", ["Writing"], "Freemium"],
    ["Textblaze", "https://textblaze.me", "N/A", ["Writing"], "Paid"],
    ["Sudowrite", "https://sudowrite.com", "N/A", ["Writing"], "Paid"],
    ["Writesonic", "https://writesonic.com", "N/A", ["Writing"], "Freemium"],
    
    # AI Model
    ["Rendernet.ai", "https://rendernet.ai", "N/A", ["AI Model"], "N/A"],
    ["Glambase App", "https://glambase.app", "N/A", ["AI Model"], "N/A"],
    ["APO8", "N/A", "N/A", ["AI Model"], "N/A"],
    ["Deepmode", "https://deepmode.ai", "N/A", ["AI Model"], "N/A"],
    ["AI Hentai", "N/A", "N/A", ["AI Model"], "N/A"],
    
    # Meeting
    ["Tldv", "https://tldv.io", "N/A", ["Meeting"], "Freemium"],
    ["Krisp", "https://krisp.ai", "N/A", ["Meeting"], "Freemium"],
    ["Otter", "https://otter.ai", "N/A", ["Meeting"], "Freemium"],
    ["Avoma", "https://avoma.com", "N/A", ["Meeting"], "Freemium"],
    ["Fireflies", "https://fireflies.ai", "N/A", ["Meeting"], "Freemium"],
    
    # Chatbot
    ["Poe", "https://poe.com", "N/A", ["Chatbot"], "Free"],
    ["Claude", "https://claude.ai", "N/A", ["Chatbot"], "Free"],
    ["Gemini", "N/A", "N/A", ["Chatbot"], "N/A"],
    ["ChatGPT", "https://chat.openai.com", "N/A", ["Chatbot", "Ideas"], "Freemium"],
    ["HuggingChat", "https://huggingface.co/chat", "N/A", ["Chatbot"], "Free"],
    
    # Automation
    ["Phrasee", "https://phrasee.co", "N/A", ["Automation"], "Paid"],
    ["Outreach", "https://outreach.io", "N/A", ["Automation"], "Paid"],
    ["ClickUp", "https://clickup.com", "N/A", ["Automation"], "Freemium"],
    ["Drift", "https://drift.com", "N/A", ["Automation"], "Freemium"],
    ["Emplifi", "https://emplifi.io", "N/A", ["Automation"], "Paid"],
    
    # UI/UX
    ["Galileo AI", "https://galileoai.com", "N/A", ["UI/UX"], "Paid"],
    ["Khroma", "https://khroma.co", "N/A", ["UI/UX"], "Free"],
    ["Uizard", "https://uizard.io", "N/A", ["UI/UX"], "Freemium"],
    ["Visily", "https://visily.ai", "N/A", ["UI/UX"], "Freemium"],
    ["VisualEyes", "https://visualeyes.design", "N/A", ["UI/UX"], "Paid"],
    
    # Image
    ["Dzine", "N/A", "N/A", ["Image"], "N/A"],
    ["Freepik", "https://freepik.com", "N/A", ["Image"], "Freemium"],
    ["Phygital+", "N/A", "N/A", ["Image"], "N/A"],
    ["Stockimg.ai", "https://stockimg.ai", "N/A", ["Image"], "Paid"],
    ["Bing Create", "https://bing.com/create", "N/A", ["Image"], "Free"],
    
    # Design
    ["Looka", "https://looka.com", "N/A", ["Design"], "Freemium"],
    ["Clipdrop", "https://clipdrop.co", "N/A", ["Design"], "Freemium"],
    ["Autodraw", "https://autodraw.com", "N/A", ["Design"], "Free"],
    ["Vance AI", "https://vanceai.com", "N/A", ["Design"], "Paid"],
    ["Designs AI", "https://designs.ai", "N/A", ["Design"], "Paid"],
    
    # Video
    ["Pictory", "https://pictory.ai", "N/A", ["Video"], "Freemium"],
    ["HeyGen", "https://heygen.com", "N/A", ["Video"], "Paid"],
    ["Nullface.ai", "N/A", "N/A", ["Video"], "N/A"],
    ["Decohere", "https://decohere.ai", "N/A", ["Video"], "Paid"],
    ["Synthesia", "https://synthesia.io", "N/A", ["Video"], "Paid"],
    
    # Blog Writer
    ["Katteb", "https://katteb.com", "N/A", ["Blog Writer"], "Freemium"],
    ["Reword", "https://reword.ai", "N/A", ["Blog Writer"], "Paid"],
    ["Elephas", "https://elephas.app", "N/A", ["Blog Writer"], "Paid"],
    ["Junia AI", "https://junia.ai", "N/A", ["Blog Writer"], "Freemium"],
    ["Journalist AI", "https://journalistai.com", "N/A", ["Blog Writer"], "Freemium"],
    
    # Marketing
    ["AdCopy", "https://adcopy.ai", "N/A", ["Marketing"], "Paid"],
    ["Predis AI", "https://predis.ai", "N/A", ["Marketing"], "Paid"],
    ["Howler AI", "https://howler.ai", "N/A", ["Marketing"], "Paid"],
    ["Bardeen AI", "https://bardeen.ai", "N/A", ["Marketing"], "Freemium"],
    ["AdCreative", "https://adcreative.ai", "N/A", ["Marketing"], "Paid"],
    
    # Twitter
    ["Metricool", "https://metricool.com", "N/A", ["Twitter"], "Freemium"],
    ["Postwise", "https://postwise.ai", "N/A", ["Twitter"], "Paid"],
    ["Tribescaler", "https://tribescaler.com", "N/A", ["Twitter"], "Paid"],
    ["TweetHunter", "https://tweethunter.io", "N/A", ["Twitter"], "Paid"],
    ["Typefully", "https://typefully.com", "N/A", ["Twitter"], "Freemium"],
    
    # Code Generator
    ["Codeium", "https://codeium.com", "N/A", ["Code Generator"], "Free"],
    ["Codex", "https://openai.com/codex", "N/A", ["Code Generator"], "Paid"],
    ["Warp AI", "https://warpdotai.com", "N/A", ["Code Generator"], "Freemium"],
    ["CodeWP", "https://codewp.ai", "N/A", ["Code Generator"], "Paid"],
    ["Replit Ghostwriter", "https://replit.com", "N/A", ["Code Generator"], "Paid"],
    
    # Speech to Text
    ["Fluently AI", "N/A", "N/A", ["Speech to Text"], "N/A"],
    ["Cockatoo AI", "N/A", "N/A", ["Speech to Text"], "N/A"],
    ["Whisper", "https://openai.com/whisper", "N/A", ["Speech to Text"], "Free"],
    ["Otter", "https://otter.ai", "N/A", ["Speech to Text", "Meeting"], "Freemium"],
    ["SpeechPlus AI", "N/A", "N/A", ["Speech to Text"], "N/A"],
    
    # AI Detector
    ["GPTZero", "https://gptzero.me", "N/A", ["AI Detector"], "Freemium"],
    ["Copyleaks", "https://copyleaks.com", "N/A", ["AI Detector"], "Paid"],
    ["BypassGPT", "https://bypassgpt.com", "N/A", ["AI Detector"], "Free"],
    ["Grammarly", "https://grammarly.com", "N/A", ["AI Detector"], "Freemium"],
    
    # Voice
    ["Udio", "N/A", "N/A", ["Voice"], "N/A"],
    ["Speech AI", "N/A", "N/A", ["Voice"], "N/A"],
    ["VEED.io", "https://veed.io", "N/A", ["Voice"], "Freemium"],
    ["ElevenLabs", "https://elevenlabs.io", "N/A", ["Voice"], "Paid"],
    
    # Web3
    ["Alva", "N/A", "N/A", ["Web3"], "N/A"],
    ["Alethea", "https://alethea.ai", "N/A", ["Web3"], "Freemium"],
    ["Adot AI", "N/A", "N/A", ["Web3"], "N/A"],
    ["Spice AI", "https://spiceai.com", "N/A", ["Web3"], "Freemium"],
    ["LIKN", "N/A", "N/A", ["Web3"], "N/A"],
]

    
    for tool in tools:
        add_tool(tool)

def verify_and_update_tools():
    """Verify existing tools and update or remove them"""
    conn = sqlite3.connect('ai_tools.db')
    c = conn.cursor()
    
    try:
        # Get tools with N/A or empty URLs
        c.execute("SELECT id, tool_name, url FROM ai_tools WHERE url = 'N/A' OR url = ''")
        tools_to_update = c.fetchall()
        
        updated_count = 0
        removed_count = 0
        
        # Create change log
        change_log = []
        change_log.append(f"Change Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        change_log.append("=" * 50 + "\n")
        
        for tool_id, tool_name, current_url in tools_to_update:
            change_log.append(f"\nChecking: {tool_name} (Current URL: {current_url})")
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a tool URL verifier. Return ONLY the URL or RETRY."},
                        {"role": "user", "content": f"Find the official website URL for the AI tool named '{tool_name}'. Return ONLY the URL if found, or 'RETRY' if unsure."}
                    ],
                    temperature=0.1
                )
                
                result = response.choices[0].message.content.strip()
                
                if result.startswith(('http://', 'https://')):
                    # Update tool with new URL
                    try:
                        c.execute("UPDATE ai_tools SET url = ? WHERE id = ?", (result, tool_id))
                        conn.commit()  # Commit immediately after update
                        
                        # Verify the update
                        c.execute("SELECT url FROM ai_tools WHERE id = ?", (tool_id,))
                        updated_url = c.fetchone()[0]
                        if updated_url == result:
                            updated_count += 1
                            change_log.append(f"✅ UPDATED: New URL - {result}")
                        else:
                            change_log.append(f"⚠️ UPDATE FAILED: Database update failed")
                    except Exception as e:
                        change_log.append(f"⚠️ UPDATE ERROR: {str(e)}")
                        conn.rollback()
                else:
                    change_log.append(f"⚠️ SKIPPED: No valid URL found - {result}")
                
            except Exception as e:
                error_msg = str(e)
                change_log.append(f"⚠️ ERROR: {error_msg}")
                continue
        
    finally:
        conn.commit()
        conn.close()
    
    # Save change log to file
    log_file = "tool_updates.log"
    change_log.append(f"\nSummary:")
    change_log.append(f"- Updated: {updated_count} tools")
    change_log.append(f"- Removed: {removed_count} tools")
    change_log.append("\n" + "=" * 50 + "\n")
    
    with open(log_file, "a", encoding='utf-8') as f:
        f.write("\n".join(change_log))
    
    return updated_count, removed_count, "\n".join(change_log)

def main():
    st.set_page_config(page_title="AI Tool Repository", layout="wide")
    st.title("🤖 AI Tool Repository")

    # Initialize and migrate database
    init_db()
    migrate_db()
    
    # Check if we need to populate initial tools
    data = get_tools_data()
    if data.empty:
        populate_initial_tools()

    # Load data once at the beginning
    data = get_tools_data()

    # Add metrics and chart at the top
    if not data.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Count only tools with valid URLs
            valid_tools = data[data['URL'].str.lower() != 'n/a']
            total_tools = len(valid_tools)
            
            # Get unique domains by splitting comma-separated values and cleaning
            all_domains = set()
            for domains in data['Domain'].str.split(','):
                if domains is not None:  # Handle NULL values
                    cleaned_domains = {d.strip() for d in domains if d.strip() and d.strip().lower() != 'n/a'}
                    all_domains.update(cleaned_domains)
            
            # Remove any invalid domain names
            all_domains = {d for d in all_domains if d in DOMAIN_CATEGORIES}
            
            col1.metric(
                label="🎯 Total AI Tools",
                value=len(data),  # Simply show total count
                delta=f"Across {len(all_domains)} domains",
                delta_color="normal"
            )
            
            # Show domain count separately
            col1.metric(
                label="🌐 Domains Covered",
                value=len(all_domains),
                delta=None
            )
            
        with col2:
            # Create domain distribution chart - handle multiple domains
            domain_counts = pd.DataFrame([
                domain.strip()
                for domains in valid_tools['Domain'].str.split(',')
                if domains is not None
                for domain in domains
                if domain.strip() in DOMAIN_CATEGORIES  # Only count valid domains
            ]).value_counts().reset_index()
            domain_counts.columns = ['Domain', 'Count']
            
            fig = px.bar(
                domain_counts,
                x='Domain',
                y='Count',
                title='Active AI Tools by Domain',
                color='Count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()  # Add a visual separator

    # Admin authentication
    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False

    with st.sidebar:
        st.subheader("👤 Admin Login")
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if user_id == st.secrets["admin"]["username"] and password == st.secrets["admin"]["password"]:
                st.session_state.admin_auth = True
                st.success("Successfully logged in!")
            else:
                st.error("Invalid credentials")

        if st.session_state.admin_auth:
            st.sidebar.subheader("🔧 Admin Panel")
            
            # Download button
            data = get_tools_data()
            if not data.empty:
                csv = get_csv_download_link(data)
                st.sidebar.download_button(
                    label="📥 Download Tools List",
                    data=csv,
                    file_name="ai_tools.csv",
                    mime="text/csv",
                )
            
            # Update URLs button
            if st.sidebar.button("🔄 Update URLs"):
                with st.spinner("Verifying and updating tools..."):
                    updated, removed, change_log = verify_and_update_tools()
                    
                    if updated > 0 or removed > 0:
                        # Store results in session state
                        st.session_state.update_results = {
                            'updated': updated,
                            'removed': removed,
                            'change_log': change_log,
                            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
                        }
                        # Force data reload
                        st.session_state.filtered_tools = get_tools_data()
                        st.rerun()
                    else:
                        st.info("No tools needed updating.")

            # Display update results if they exist
            if hasattr(st.session_state, 'update_results'):
                st.success(f"Updated {st.session_state.update_results['updated']} tools and removed {st.session_state.update_results['removed']} discontinued tools.")
                
                # Show change log in expandable section
                with st.expander("View Change Log", expanded=True):
                    st.code(st.session_state.update_results['change_log'])
                
                # Add download button for the log
                st.download_button(
                    label="📥 Download Change Log",
                    data=st.session_state.update_results['change_log'],
                    file_name=f"tool_updates_{st.session_state.update_results['timestamp']}.log",
                    mime="text/plain"
                )
                
                # Add button to clear the update results
                if st.button("Clear Update Report"):
                    del st.session_state.update_results
                    st.rerun()

            # Manual tool addition form
            with st.sidebar.form("add_tool_form"):
                st.write("Add New Tool")
                tool_name = st.text_input("Tool Name")
                url = st.text_input("URL")
                github_url = st.text_input("GitHub URL (optional)")
                domain = st.selectbox("Domain", DOMAIN_CATEGORIES)
                pricing = st.selectbox("Pricing", ["Free", "Freemium", "Paid", "Enterprise"])
                
                if st.form_submit_button("Add Tool"):
                    if tool_name and url:
                        tool_data = [tool_name, url, github_url or 'N/A', [domain], pricing]
                        if add_tool(tool_data):
                            st.success("Tool added successfully!")
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Tool Name and URL are required!")

            # Auto-find tools button
            if st.sidebar.button("🔄 Auto-Find New Tools"):
                with st.spinner("Searching for new AI tools (max 60 seconds)..."):
                    data = get_tools_data()
                    new_tools = search_for_new_tools(data.to_dict())
                    if new_tools:
                        new_tools_list = [line.split(',') for line in new_tools.split('\n') if line.strip()]
                        added_count = 0
                        for tool in new_tools_list[:10]:
                            if len(tool) == 5:  # Ensure we have all required fields
                                if add_tool(tool):
                                    added_count += 1
                        if added_count > 0:
                            st.success(f"Added {added_count} new tools successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("No valid new tools could be added.")
                    else:
                        st.info("No new tools found at this time.")

    # Load and display data
    data = get_tools_data()

    # Tool recommendation section
    st.subheader("🔍 Find the Perfect AI Tool")
    user_query = st.text_area(
        "Describe your needs:",
        placeholder="Example: I need an AI tool to help me write marketing copy..."
    )
    
    # Get unique domains from the database for the dropdown
    available_domains = ["All"] + sorted(data["Domain"].unique().tolist())
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_domain = st.selectbox("Filter by Domain", available_domains)
    
    # Store the filtered dataframe in session state to share between sections
    if "filtered_tools" not in st.session_state:
        st.session_state.filtered_tools = data.copy()
    
    if st.button("Find Tools"):
        if user_query:
            with st.spinner("Analyzing your needs..."):
                # Handle multiple domains in filtering
                filtered_data = data
                if selected_domain != "All":
                    filtered_data = data[
                        data['Domain'].str.split(',').apply(
                            lambda domains: selected_domain.strip().lower() in [d.strip().lower() for d in domains]
                        )
                    ]
                
                if filtered_data.empty:
                    st.warning(f"""Sorry, no tools found for domain: {selected_domain}
                    Available domains in database: {', '.join(data['Domain'].unique())}""")
                    st.session_state.filtered_tools = pd.DataFrame(columns=SHEET_COLUMNS)
                else:
                    recommendations = recommend_tools(user_query, filtered_data)
                    if recommendations:
                        st.write(recommendations)
                        
                        # Show suggested domains if any
                        if hasattr(st.session_state, 'suggested_domains'):
                            st.info(f"🎯 Suggested domains for your need: {', '.join(st.session_state.suggested_domains)}")
        else:
            st.warning("Please describe your needs first.")

    # Display tool database
    st.subheader("📚 Available AI Tools")
    if not data.empty:
        # Add filters using actual domains from database
        col1, col2 = st.columns([1, 1])
        with col1:
            domain_filter = st.multiselect("Filter by Domain", sorted(data["Domain"].unique().tolist()))
        with col2:
            pricing_filter = st.multiselect("Filter by Pricing", sorted(data["Pricing"].unique().tolist()))

        # Use the filtered tools from the search if available
        display_df = st.session_state.filtered_tools.copy()
        
        # Apply additional filters if selected
        if domain_filter:
            display_df = display_df[display_df["Domain"].str.strip().str.lower().isin([d.strip().lower() for d in domain_filter])]
        if pricing_filter:
            display_df = display_df[display_df["Pricing"].str.strip().str.lower().isin([p.strip().lower() for p in pricing_filter])]

        if display_df.empty:
            st.info("No tools match the current filters or search criteria.")
        else:
            # Rename id column to '#' and show it
            if 'id' in display_df.columns:
                display_df = display_df.rename(columns={'id': '#'})
                # Reorder columns to show ID first
                cols = ['#'] + [col for col in display_df.columns if col != '#']
                display_df = display_df[cols]
            
            # Pagination
            items_per_page = 25
            total_pages = len(display_df) // items_per_page + (1 if len(display_df) % items_per_page > 0 else 0)
            
            # Initialize pagination in session state if not exists
            if "current_page" not in st.session_state:
                st.session_state.current_page = 1
            
            current_page = st.session_state.current_page
            start_idx = (current_page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(display_df))
            
            # Show page info
            st.caption(f"Showing {start_idx + 1} to {end_idx} of {len(display_df)} tools")
            
            # Display paginated dataframe
            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )
            
            # Add page navigation buttons
            col1, col2, col3, col4 = st.columns([3, 1, 1, 3])
            with col2:
                if current_page > 1:
                    if st.button("◀ Previous"):
                        st.session_state.current_page -= 1
                        st.rerun()
            with col3:
                if current_page < total_pages:
                    if st.button("Next ▶"):
                        st.session_state.current_page += 1
                        st.rerun()
    else:
        st.info("No tools available in the database yet.")

    # Add a button to reset filters and show all tools
    if st.button("Show All Tools"):
        st.session_state.filtered_tools = data.copy()
        st.rerun()

if __name__ == "__main__":
    main()
