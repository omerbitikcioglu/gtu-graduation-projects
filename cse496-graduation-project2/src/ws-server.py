import asyncio
import librosa
import websockets
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sample_rate = 44100

try:
    # Load model
    model = load_model('tl_ser.h5')

except Exception as e:
    print(e)

f = pd.read_csv('features_tl.csv')
encoder = OneHotEncoder()
y = f['labels'].values
encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
scaler = StandardScaler()


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sr, pitch_factor):
    return librosa.effects.pitch_shift(data, sr, pitch_factor)


def extract_features(data):
    # Zero Crossing Rate
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(audio):
    # without augmentation
    res1 = extract_features(audio)
    result = np.array(res1)

    # data with noise
    noise_data = noise(audio)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(audio, rate=0.8)
    data_stretch_pitch = pitch(new_data, sample_rate, pitch_factor=0.7)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result


# Function to process the selected audio file
def process_audio(audio):
    features = get_features(audio)

    # Apply the StandardScaler to the features
    features = scaler.fit_transform(features)

    # Make the features compatible with the model
    features = np.expand_dims(features, axis=2)

    # Use the model to make predictions
    prediction = model.predict(features)

    # Use the inverse_transform method to convert the one-hot encoded predictions back to emotions
    y_pred = encoder.inverse_transform(prediction)

    # Return the most occurred result
    flatten_pred = [item for sublist in y_pred for item in sublist]
    max_pred = max(set(flatten_pred), key=flatten_pred.count)

    return max_pred


async def classify_audio(websocket, path):
    while True:
        try:
            # Receive audio data from the client
            data = await websocket.recv()
            # print(data)

            # Convert the binary data to a float array
            if isinstance(data, bytes):
                audio = np.frombuffer(data, dtype=np.float32)
                # print(audio)
                classification = process_audio(audio)
                await websocket.send(classification)

        except Exception as e:
            print(f"Error message: {e}")
            # If an error occurs, log the error and close the WebSocket
            print("An error occurred. Closing WebSocket.")
            await websocket.close()


# Start the WebSocket server
start_server = websockets.serve(classify_audio, "localhost", 8080)

# Run the server forever
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
