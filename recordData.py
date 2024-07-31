#Codigo para recibir grabacion de audio


#Librerias
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt



# Configurar la salida de sonido para el modelo
# Audio is resampled to 16 kHz mono.
fs = 16000  # Sample rate
Nchannel = 1 # Number of channels (1 = mono)
chunk_duration = 3  # Desired chunk duration in seconds

# Arreglo de muestras de informacion captadas
buffer_array = []


def show_graph(buffer_array):
  for index, buf in enumerate(buffer_array):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(buf)
    ax1.set_title(str(index*chunk_duration) + " - " + str(index*chunk_duration+chunk_duration))
    plt.show()

# FUNCION QUE SE LLAMA MIENTRAS SE GRABA EL AUDIO
def audio_callback(indata, frames, time, status):
  """Callback function for processing incoming audio data."""
  # Get audio data as a NumPy array
  audio_data = indata[:, 0]


  

  # Collect data from current chunk
  global buffer
  buffer = np.concatenate((buffer, audio_data))  # Concatenate to global buffer

  # Check if chunk duration is reached
  if len(buffer) / fs >= chunk_duration:
    # Process the collected chunk (replace with your logic)
    process_chunk(buffer)
    buffer_array.append(buffer)
    
  

    # Clear buffer for next chunk
    buffer = np.array([])



# PROCESAMIENTO DE INFORMACION POR CHUNK
def process_chunk(chunk_data):
  """Function to process the captured audio chunk."""
  # Perform your desired processing on the chunk data (e.g., feature extraction, classification)
  # Visualize the chunk data (example using time-domain waveform)

  """Function to analyze frequency and amplitude."""
  # Calculate FFT
  fft_data = np.fft.fft(chunk_data)

  # Frequencies (half the FFT size due to symmetry for real signals)
  frequencies = np.fft.fftfreq(len(chunk_data), d=1/fs)

  # Calculate magnitude spectrum
  magnitude = np.abs(fft_data)

  print("fft")
  print(fft_data.size)
  print("frequencies")
  print(frequencies.size)

  # Print or process frequencies and amplitudes
  print(f"Dominant frequencies (Hz): {frequencies[:5]}")


 
  
  print(np.info(chunk_data))
  print(chunk_data.shape)
  print(f"Chunk data processed (size: {len(chunk_data)} samples)")

 

# Global variable to store accumulated audio data
buffer = np.array([])

with sd.InputStream(samplerate=fs, channels=Nchannel, callback=audio_callback):
  print("Listening...")
  try:
    while True:
      # Loop indefinitely
      pass
  except KeyboardInterrupt:
    # Handle keyboard interrupt (ctrl+c) to stop recording
    print("Recording stopped.")


print(buffer_array)
show_graph(buffer_array)

print("Program ended.")