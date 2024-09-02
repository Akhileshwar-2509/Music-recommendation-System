# main.py

import requests
import base64
import pandas as pd
import spotipy
from config import CLIENT_ID, CLIENT_SECRET
from utils import hybrid_recommendations

def get_access_token(client_id, client_secret):
    client_credentials = f"{client_id}:{client_secret}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode())
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {client_credentials_base64.decode()}'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        return response.json()['access_token']
    else:
        print("Error obtaining access token.")
        exit()

def get_trending_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    playlist_tracks = sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))')

    music_data = []
    for track_info in playlist_tracks['items']:
        track = track_info['track']
        track_name = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        album_name = track['album']['name']
        album_id = track['album']['id']
        track_id = track['id']

        audio_features = sp.audio_features(track_id)[0] if track_id != 'Not available' else None
        try:
            album_info = sp.album(album_id) if album_id != 'Not available' else None
            release_date = album_info['release_date'] if album_info else None
        except:
            release_date = None

        try:
            track_info = sp.track(track_id) if track_id != 'Not available' else None
            popularity = track_info['popularity'] if track_info else None
        except:
            popularity = None

        track_data = {
            'Track Name': track_name,
            'Artists': artists,
            'Album Name': album_name,
            'Album ID': album_id,
            'Track ID': track_id,
            'Popularity': popularity,
            'Release Date': release_date,
            'Duration (ms)': audio_features['duration_ms'] if audio_features else None,
            'Explicit': track_info.get('explicit', None),
            'External URLs': track_info.get('external_urls', {}).get('spotify', None),
            'Danceability': audio_features['danceability'] if audio_features else None,
            'Energy': audio_features['energy'] if audio_features else None,
            'Key': audio_features['key'] if audio_features else None,
            'Loudness': audio_features['loudness'] if audio_features else None,
            'Mode': audio_features['mode'] if audio_features else None,
            'Speechiness': audio_features['speechiness'] if audio_features else None,
            'Acousticness': audio_features['acousticness'] if audio_features else None,
            'Instrumentalness': audio_features['instrumentalness'] if audio_features else None,
            'Liveness': audio_features['liveness'] if audio_features else None,
            'Valence': audio_features['valence'] if audio_features else None,
            'Tempo': audio_features['tempo'] if audio_features else None,
        }

        music_data.append(track_data)

    df = pd.DataFrame(music_data)
    return df

if __name__ == "__main__":
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    playlist_id = '37i9dQZF1DX76Wlfdnj7AP'
    music_df = get_trending_playlist_data(playlist_id, access_token)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    music_features = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness',
                               'Instrumentalness', 'Liveness', 'Valence', 'Tempo']].values
    music_features_scaled = scaler.fit_transform(music_features)

    input_song_name = "I'm Good (Blue)"
    recommendations = hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5)
    print(f"Hybrid recommended songs for '{input_song_name}':")
    print(recommendations)
