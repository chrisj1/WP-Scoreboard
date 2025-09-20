import sys
import requests

def usage():
    print("Usage: python scoreboard_control.py <url> <action>")
    print("Actions: add_home, add_away, remove_home, remove_away, next_period, prev_period, next_game, toggle_home_manup, toggle_away_manup")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        usage()
    url = sys.argv[1].rstrip('/')
    action = sys.argv[2]
    endpoints = {
        'add_home': '/api/add_home',
        'add_away': '/api/add_away',
        'remove_home': '/api/remove_home',
        'remove_away': '/api/remove_away',
        'next_period': '/api/next_period',
        'prev_period': '/api/prev_period',
        'next_game': '/api/next_game',
        'prev_game': '/api/prev_game',
        'reset': '/api/reset',
        'toggle_home_manup': '/api/toggle_home_manup',
        'toggle_away_manup': '/api/toggle_away_manup',
        'toggle_hide_scores': '/api/toggle_hide_scores',
        'refresh_data': '/api/refresh_data',
    }
    if action not in endpoints:
        print(f"Unknown action: {action}")
        usage()
    endpoint = endpoints[action]
    full_url = url + endpoint
    try:
        resp = requests.get(full_url)
        resp.raise_for_status()
        print(resp.json())
    except Exception as e:
        print(f"Error calling {full_url}: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
