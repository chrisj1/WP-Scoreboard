TEAM_NAME_FONT_SIZE = '6vw'
TEAM_NAME_MIN_FONT_SIZE = '40px'
SCORE_FONT_SIZE = '6vw'
SCORE_MIN_FONT_SIZE = '40px'
# Store font sizes in a global dict
FONT_SIZES = {
  'team': TEAM_NAME_FONT_SIZE,
  'score': SCORE_FONT_SIZE,
}


from flask import send_from_directory
from flask import Flask, request, jsonify, send_from_directory, make_response
import json
app = Flask(__name__)

# --- API Shortcuts for Stream Deck/Remote Control ---
# ...existing code...

# Load teams from CSV for initial values
import csv
def get_first_two_teams():
  with open('teams.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    teams = list(reader)
    return teams[0], teams[1]
t1, t2 = get_first_two_teams()
scoreboard = {
  "team1": {"name": t1['name'], "score": 0, "color": t1['home_bg'], "manup": False, "manup_expiry": None},
  "team2": {"name": t2['name'], "score": 0, "color": t2['away_bg'], "manup": False, "manup_expiry": None},
  "period": "1st",
  "hide_scores": False
}

from flask import redirect, url_for


# Live-updating home players page
@app.route('/players/home')
def players_home_live():
  import csv, os, math
  team_name = scoreboard['team1']['name']
  page = int(request.args.get('page', 1))
  csv_path = os.path.join('players', f'{team_name}.csv')
  players = []
  try:
    with open(csv_path, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        players.append(row)
  except FileNotFoundError:
    players = []
  per_page = 20
  total_pages = math.ceil(len(players) / per_page)
  start = (page - 1) * per_page
  end = start + per_page
  page_players = players[start:end]
  print('DEBUG: Loaded players:', players)
  print('DEBUG: Page players:', page_players)
  # Map CSV headers to expected keys for template
  for p in page_players:
    # Try all possible header variants
    p['first_name'] = p.get('first_name') or p.get('First Name') or p.get('first') or ''
    p['last_name'] = p.get('last_name') or p.get('Last Name') or p.get('last') or ''
    p['capnumber'] = p.get('capnumber') or p.get('Cap Number') or ''
  # Get team color and logo
  color = scoreboard['team1']['color'] if team_name == scoreboard['team1']['name'] else None
  if not color:
    try:
      color = load_teams()[team_name]['home_bg']
    except Exception:
      color = '#222'
  logo = f"{team_name.replace(' ', '').lower()}.svg"
  return render_template(
    'players.html',
    team_name=team_name,
    players=page_players,
    page=page,
    total_pages=total_pages,
    team_color=color,
    team_logo=logo
  )

# Live-updating away players page
@app.route('/players/away')
def players_away_live():
  import csv, os, math
  team_name = scoreboard['team2']['name']
  page = int(request.args.get('page', 1))
  csv_path = os.path.join('players', f'{team_name}.csv')
  players = []
  try:
    with open(csv_path, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        players.append(row)
  except FileNotFoundError:
    players = []
  per_page = 20
  total_pages = math.ceil(len(players) / per_page)
  start = (page - 1) * per_page
  end = start + per_page
  page_players = players[start:end]
  # Map CSV headers to expected keys for template
  for p in page_players:
    p['first_name'] = p.get('first_name') or p.get('First Name') or p.get('first') or ''
    p['last_name'] = p.get('last_name') or p.get('Last Name') or p.get('last') or ''
    p['capnumber'] = p.get('capnumber') or p.get('Cap Number') or ''
  color = scoreboard['team2']['color'] if team_name == scoreboard['team2']['name'] else None
  if not color:
    try:
      color = load_teams()[team_name]['away_bg']
    except Exception:
      color = '#222'
  logo = f"{team_name.replace(' ', '').lower()}.svg"
  return render_template(
    'players.html',
    team_name=team_name,
    players=page_players,
    page=page,
    total_pages=total_pages,
    team_color=color,
    team_logo=logo
  )

# Add a point to home
@app.route('/api/add_home')
def api_add_home():
  scoreboard['team1']['score'] += 1
  return ('', 204)

# Add a point to away
@app.route('/api/add_away')
def api_add_away():
  scoreboard['team2']['score'] += 1
  return ('', 204)

# Remove a point from home
@app.route('/api/remove_home')
def api_remove_home():
  scoreboard['team1']['score'] = max(0, scoreboard['team1']['score'] - 1)
  return ('', 204)

# Remove a point from away
@app.route('/api/remove_away')
def api_remove_away():
  scoreboard['team2']['score'] = max(0, scoreboard['team2']['score'] - 1)
  return ('', 204)

# Advance the period
@app.route('/api/next_period')
def api_next_period():
  periods = ['1st', '2nd', '3rd', '4th', '5th', 'Shoot-Off']
  idx = periods.index(scoreboard['period']) if scoreboard['period'] in periods else 0
  scoreboard['period'] = periods[(idx + 1) % len(periods)]
  return ('', 204)

# Go back a period
@app.route('/api/prev_period')
def api_prev_period():
  periods = ['1st', '2nd', '3rd', '4th', '5th', 'Shoot-Off']
  idx = periods.index(scoreboard['period']) if scoreboard['period'] in periods else 0
  scoreboard['period'] = periods[(idx - 1) % len(periods)]
  return ('', 204)

# Go to the next game in the schedule
@app.route('/api/next_game')
def api_next_game():
  schedule = load_schedule()
  if not schedule:
    return ('', 204)
  # Find current game index
  current = None
  for i, row in enumerate(schedule):
    if (row['home'] == scoreboard['team1']['name'] and row['away'] == scoreboard['team2']['name']):
      current = i
      break
  next_idx = (current + 1) % len(schedule) if current is not None else 0
  next_game = schedule[next_idx]
  teams = load_teams()
  scoreboard['team1']['name'] = next_game['home']
  scoreboard['team2']['name'] = next_game['away']
  scoreboard['team1']['score'] = 0
  scoreboard['team2']['score'] = 0
  scoreboard['team1']['color'] = teams[next_game['home']]['home_bg']
  scoreboard['team2']['color'] = teams[next_game['away']]['away_bg']
  scoreboard['period'] = next_game.get('period', '1st')
  return ('', 204)

# Go to the previous game in the schedule
@app.route('/api/prev_game')
def api_prev_game():
  schedule = load_schedule()
  if not schedule:
    return ('', 204)
  # Find current game index
  current = None
  for i, row in enumerate(schedule):
    if (row['home'] == scoreboard['team1']['name'] and row['away'] == scoreboard['team2']['name']):
      current = i
      break
  prev_idx = (current - 1) % len(schedule) if current is not None else 0
  prev_game = schedule[prev_idx]
  teams = load_teams()
  scoreboard['team1']['name'] = prev_game['home']
  scoreboard['team2']['name'] = prev_game['away']
  scoreboard['team1']['score'] = 0
  scoreboard['team2']['score'] = 0
  scoreboard['team1']['color'] = teams[prev_game['home']]['home_bg']
  scoreboard['team2']['color'] = teams[prev_game['away']]['away_bg']
  scoreboard['period'] = prev_game.get('period', '1st')
  return ('', 204)

# Toggle home man up
@app.route('/api/toggle_home_manup')
def api_toggle_home_manup():
  import time
  scoreboard['team1']['manup'] = not scoreboard['team1']['manup']
  scoreboard['team1']['manup_expiry'] = time.time() + 30 if scoreboard['team1']['manup'] else None
  return ('', 204)

# Toggle away man up
@app.route('/api/toggle_away_manup')
def api_toggle_away_manup():
  import time
  scoreboard['team2']['manup'] = not scoreboard['team2']['manup']
  scoreboard['team2']['manup_expiry'] = time.time() + 30 if scoreboard['team2']['manup'] else None
  return ('', 204)

# Reset scores and period
@app.route('/api/reset')
def api_reset():
  scoreboard['team1']['score'] = 0
  scoreboard['team2']['score'] = 0
  scoreboard['period'] = '1st'
  return ('', 204)

@app.route('/api/toggle_hide_scores')
def api_toggle_hide_scores():
  value = request.args.get('value')
  if value is not None:
    scoreboard['hide_scores'] = value.lower() == 'true'
  else:
    scoreboard['hide_scores'] = not scoreboard['hide_scores']
  return ('', 204)

@app.route('/api/refresh_data')
def api_refresh_data():
  global _teams_cache, _schedule_cache
  _teams_cache = None
  _schedule_cache = None
  # Force reload by calling the functions
  load_teams()
  load_schedule()
  return ('', 204)

# API to get current font sizes
@app.route('/font_sizes')
def get_font_sizes():
  return jsonify(FONT_SIZES)

# API to update font sizes
@app.route('/set_font_size', methods=['POST'])
def set_font_size():
  data = request.json
  t = data.get('type')
  val = data.get('value')
  if t in FONT_SIZES and val:
    FONT_SIZES[t] = val
  return ('', 204)


# Serve team logos from the logos/ directory (must be after app is defined)
@app.route('/logos/<path:filename>')
def serve_logo(filename):
  return send_from_directory('logos', filename)



# --- Overlay Page ---
from flask import render_template

# --- Players View ---
import math
@app.route('/players/<team_name>')
def players_view(team_name):
  import csv
  import os
  page = int(request.args.get('page', 1))
  csv_path = os.path.join('players', f'{team_name}.csv')
  players = []
  try:
    with open(csv_path, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        players.append(row)
  except FileNotFoundError:
    players = []
  per_page = 20
  total_pages = math.ceil(len(players) / per_page)
  start = (page - 1) * per_page
  end = start + per_page
  page_players = players[start:end]
  # Ensure keys for template
  # Map CSV headers to expected keys for template
  for p in page_players:
    p['first_name'] = p.get('first_name') or p.get('First Name') or p.get('first') or ''
    p['last_name'] = p.get('last_name') or p.get('Last Name') or p.get('last') or ''
    p['capnumber'] = p.get('capnumber') or p.get('Cap Number') or ''
  # Try to get color from teams.csv, fallback to #222
  try:
    color = load_teams()[team_name]['home_bg']
  except Exception:
    color = '#222'
  logo = f"{team_name.replace(' ', '').lower()}.svg"
  return render_template(
    'players.html',
    team_name=team_name,
    players=page_players,
    page=page,
    total_pages=total_pages,
    team_color=color,
    team_logo=logo
  )

@app.route("/overlay")
def overlay():
    team1_logo = f"{scoreboard['team1']['name'].replace(' ', '').lower()}.svg"
    team2_logo = f"{scoreboard['team2']['name'].replace(' ', '').lower()}.svg"
    period = scoreboard['period']
    team1_color = scoreboard['team1']['color']
    team2_color = scoreboard['team2']['color']
    return render_template(
        "overlay.html",
        team1_logo=team1_logo,
        team2_logo=team2_logo,
        period=period,
        team1_color=team1_color,
        team2_color=team2_color,
        team1_name=scoreboard['team1']['name'],
        team2_name=scoreboard['team2']['name'],
        team1_score=scoreboard['team1']['score'],
        team2_score=scoreboard['team2']['score']
    )

# --- Config Page ---
import csv

# Cache teams and schedule data to avoid repeated file reads
_teams_cache = None
_schedule_cache = None

def load_teams():
    global _teams_cache
    if _teams_cache is None:
        _teams_cache = {}
        with open('teams.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                _teams_cache[row['name']] = row
    return _teams_cache

def load_schedule():
    global _schedule_cache
    if _schedule_cache is None:
        _schedule_cache = []
        try:
            with open('schedule.csv', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    _schedule_cache.append(row)
        except FileNotFoundError:
            pass
    return _schedule_cache

@app.route("/config")
def config():
    teams = load_teams()
    schedule = load_schedule()
    def options_html(selected):
        return "\n".join(
            f"<option value='{t}'{' selected' if t == selected else ''}>{t}</option>"
            for t in teams
        )
    def schedule_options_html():
        return "\n".join(
            f"<option value='{i}'>{row['home']} vs {row['away']} ({row.get('start_time','')})</option>"
            for i, row in enumerate(schedule)
        )
    return f'''<!DOCTYPE html>
<html>
<head>
  <title>Scoreboard Config</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #222; color: white; padding: 20px; }}
    label {{ display: block; margin-top: 16px; }}
    select, input[type=text] {{ margin-left: 10px; font-size: 16px; }}
  </style>
  <script>
    const teamData = {json.dumps(teams)};
    const schedule = {json.dumps(schedule)};
    async function updateTeam(which) {{
      const team = document.getElementById(which).value;
      let color = (which === 'team1') ? teamData[team]['home_bg'] : teamData[team]['away_bg'];
      document.getElementById(which + 'color').value = color;
      await fetch('/set', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ team: which, name: team, color: color }})
      }});
    }}
    async function updateColor(which) {{
      const team = document.getElementById(which).value;
      const color = document.getElementById(which + 'color').value;
      await fetch('/set', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ team: which, name: team, color: color }})
      }});
    }}
    async function updatePeriod() {{
      const period = document.getElementById('period').value;
      await fetch('/set', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ period: period }})
      }});
    }}
    async function updateScore(which) {{
      const val = parseInt(document.getElementById(which + 'score').value);
      await fetch('/set', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ team: which, score: val }})
      }});
    }}
    async function updateManup(which) {{
      const checked = document.getElementById(which + 'manup').checked;
      await fetch('/set', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ team: which, manup: checked }})
      }});
    }}
    async function updateHideScores() {{
      const checked = document.getElementById('hidescores').checked;
      await fetch('/api/toggle_hide_scores?value=' + checked);
    }}
    async function refreshData() {{
      await fetch('/api/refresh_data');
      // Reload the page to get fresh data
      window.location.reload();
    }}
    async function applyPreset() {{
      const idx = document.getElementById('preset').value;
      if (idx === "") return;
      const preset = schedule[parseInt(idx)];
      if (preset) {{
        document.getElementById('team1').value = preset.home;
        document.getElementById('team2').value = preset.away;
        document.getElementById('period').value = preset.period || '1st';
        updateTeam('team1');
        updateTeam('team2');
        updatePeriod();
        // Trigger onchange for both team dropdowns so manual changes work as expected
        document.getElementById('team1').dispatchEvent(new Event('change'));
        document.getElementById('team2').dispatchEvent(new Event('change'));
      }}
    }}
    // Font size controls
    async function updateFontSize(type) {{
      const val = document.getElementById(type + 'FontSize').value;
      await fetch('/set_font_size', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ type: type, value: val }})
      }});
    }}
    // Poll state and sync manup checkboxes
    function pollState() {{
      fetch('/state')
        .then(res => res.json())
        .then(data => {{
          const t1 = document.getElementById('team1manup');
          const t2 = document.getElementById('team2manup');
          const hide = document.getElementById('hidescores');
          if (t1) t1.checked = !!data.team1.manup;
          if (t2) t2.checked = !!data.team2.manup;
          if (hide) hide.checked = !!data.hide_scores;
        }})
        .catch(e => {{}});
      setTimeout(pollState, 1000);
    }}
    window.onload = function() {{
      pollState();
    }};
  </script>
</head>
<body>
  <label>Team 1 Man-Up:
    <input type="checkbox" id="team1manup" onchange="updateManup('team1')">
  </label>
  <label>Team 2 Man-Up:
    <input type="checkbox" id="team2manup" onchange="updateManup('team2')">
  </label>
  <label>Hide Scores:
    <input type="checkbox" id="hidescores" onchange="updateHideScores()">
  </label>
  <label>
    <button onclick="refreshData()" style="margin-top: 16px; padding: 8px 16px; font-size: 14px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Refresh Teams & Schedule Data</button>
  </label>
  <label>Team Name Text Size:
    <input id="teamFontSize" type="text" value="2.8vw" onchange="updateFontSize('team')">
  </label>
  <label>Score Text Size:
    <input id="scoreFontSize" type="text" value="7vw" onchange="updateFontSize('score')">
  </label>
  <label>Preset Match:
    <select id="preset" onchange="applyPreset()">
      <option value="">-- Select a match --</option>
      {schedule_options_html()}
    </select>
  </label>
  <label>Team 1:
    <select id="team1" onchange="updateTeam('team1')">
      {options_html(scoreboard['team1']['name'])}
    </select>
    <input id="team1color" type="color" value="{scoreboard['team1']['color']}" onchange="updateColor('team1')">
  </label>
  <label>Team 1 Score: <input id="team1score" type="number" min="0" value="{scoreboard['team1']['score']}" onchange="updateScore('team1')"></label>
  <label>Team 2:
    <select id="team2" onchange="updateTeam('team2')">
      {options_html(scoreboard['team2']['name'])}
    </select>
    <input id="team2color" type="color" value="{scoreboard['team2']['color']}" onchange="updateColor('team2')">
  </label>
  <label>Team 2 Score: <input id="team2score" type="number" min="0" value="{scoreboard['team2']['score']}" onchange="updateScore('team2')"></label>
  <label>Period: 
    <select id="period" onchange="updatePeriod()">
      <option value="1st" {'selected' if scoreboard['period']=='1st' else ''}>1st</option>
      <option value="2nd" {'selected' if scoreboard['period']=='2nd' else ''}>2nd</option>
      <option value="3rd" {'selected' if scoreboard['period']=='3rd' else ''}>3rd</option>
      <option value="4th" {'selected' if scoreboard['period']=='4th' else ''}>4th</option>
      <option value="5th" {'selected' if scoreboard['period']=='5th' else ''}>5th</option>
      <option value="Shoot-off" {'selected' if scoreboard['period']=='Shoot-off' else ''}>Shoot-off</option>
    </select>
  </label>
</body>
</html>
'''

# --- API ---
@app.route("/state")
def state():
  import time
  now = time.time()
  # Check expiry for both teams
  for t in ["team1", "team2"]:
      expiry = scoreboard[t].get("manup_expiry")
      if expiry and expiry <= now:
          scoreboard[t]["manup"] = False
          scoreboard[t]["manup_expiry"] = None
  return jsonify(scoreboard)

@app.route("/set", methods=["POST"])
def set_score():
    data = request.json
    if "team" in data and data["team"] in scoreboard:
        team = data["team"]
        if "score" in data:
            scoreboard[team]["score"] = data["score"]
        if "name" in data:
            scoreboard[team]["name"] = data["name"]
        if "color" in data:
            scoreboard[team]["color"] = data["color"]
    if "manup" in data:
      import time
      if data["manup"]:
        scoreboard[team]["manup"] = True
        scoreboard[team]["manup_expiry"] = time.time() + 30
      else:
        scoreboard[team]["manup"] = False
        scoreboard[team]["manup_expiry"] = None
    if "period" in data:
        scoreboard["period"] = data["period"]
    return jsonify(scoreboard)

@app.route("/add", methods=["POST"])
def add_point():
    data = request.json
    team = data.get("team")
    if team in scoreboard:
        scoreboard[team]["score"] += 1
    import time
    now = time.time()
    # Check expiry for both teams
    for t in ["team1", "team2"]:
        expiry = scoreboard[t].get("manup_expiry")
        if expiry and expiry <= now:
            scoreboard[t]["manup"] = False
            scoreboard[t]["manup_expiry"] = None
    return jsonify(scoreboard)

@app.route("/reset", methods=["POST"])
def reset():
    for t in ["team1", "team2"]:
        scoreboard[t]["score"] = 0
    scoreboard["period"] = "1st"
    return jsonify(scoreboard)

@app.route("/intermission")
def intermission():
    team1_logo = f"{scoreboard['team1']['name'].replace(' ', '').lower()}.svg"
    team2_logo = f"{scoreboard['team2']['name'].replace(' ', '').lower()}.svg"
    period = scoreboard['period']
    team1_score = scoreboard['team1']['score']
    team2_score = scoreboard['team2']['score']
    return render_template(
        "intermission.html",
        team1_logo=team1_logo,
        team2_logo=team2_logo,
        period=period,
        team1_score=team1_score,
        team2_score=team2_score
    )

@app.route("/schedule")
def schedule_view():
    import csv
    from collections import defaultdict
    # Load teams and their colors
    teams = {}
    with open('teams.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            teams[row['name']] = row
    # Load schedule and group by day
    days = defaultdict(list)
    with open('schedule.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            day = row['start_time'].split()[0]
            days[day].append(row)
    days_list = sorted(days.keys())
    return render_template('schedule.html', days=days, days_list=days_list, teams=teams)

if __name__ == "__main__":
    app.run(port=5000)
