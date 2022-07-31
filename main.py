"""
Author: Mohammad Ramezani
Created: July 31, 2022
"""

import glob
import os
import winsound
from os import path
import numpy as np
import pretty_midi
from scipy import sparse
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from datetime import datetime

# For Music File Transcription
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio


DIRECTORY = 'D:/Business/Idea_Music/Data/Original_Data/IMSLP/'
global COMPLEXITY_LEVEL_NUMBER
WINDOWS_WIDTH = 500


def complexity_class_number_computation(desired_path):
    os.walk(desired_path)
    cnt = 0
    for x in os.walk(desired_path):
        cnt += 1

    print(f'\nNumber of complexity levels =', cnt - 1)
    return cnt - 1


def each_level_track_number(folder_path, complex_level):
    sample_complex = []
    for level in range(1, complex_level + 1):
        pth = folder_path + "{:02d}".format(level) + '/'
        initial_count = 0
        for p in os.listdir(pth):
            if os.path.isfile(os.path.join(pth, p)) and p[-3:] == 'mp3':
                initial_count += 1
        sample_complex.append(initial_count)

    print('Number of samples in each complexity level =', sample_complex)
    return sample_complex


def mp3_to_midi(mp3_name, mid_name):
    (audio, _) = load_audio(mp3_name, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cpu')  # 'cuda' | 'cpu'
    transcribed_dict = transcriptor.transcribe(audio, mid_name)


def convert_mp3_to_midi(directory):
    # Mp3 files searching and converting them to MIDI
    print(f"\nPlease wait! \nIt is checking for Mp3 to MIDI file conversion...")
    all_mp3_number = 0
    directory_mp3 = directory + '**/*'
    for file in glob.glob(directory_mp3):
        mid_filename = file[:-3] + 'mid'
        if not path.exists(mid_filename) and file[-3:] == 'mp3':
            print(file)
            all_mp3_number += 1
            mp3_filename = file
            mp3_to_midi(mp3_filename, mid_filename)
    print('Total MP3 to MIDI transcription =', all_mp3_number)


def midi_to_pianoroll():

    print(f"\nPlease wait! \nThe pianorolls are being created...\n ")

    database_piano_roll = []
    for i in range(1, COMPLEXITY_LEVEL_NUMBER + 1):
        print(i)
        temp_dir = DIRECTORY + "{:02d}".format(i)
        dir_list = os.listdir(temp_dir)
        # dir_list = [dir_list[0]]  # to choose a desired sample
        j = 0
        for filename in dir_list:
            if filename[-3:] == 'mid':
                j += 1
                midi_filename = temp_dir + '/' + filename
                raw_midi_data = pretty_midi.PrettyMIDI(midi_filename)
                piano_roll = raw_midi_data.get_piano_roll(fs=100)
                piano_roll = piano_roll.astype(np.float16)
                sparse_piano_roll = sparse.csr_matrix(piano_roll)
                database_piano_roll.append([i, j, sparse_piano_roll])

    np.save('database_pianoroll.npy', database_piano_roll, allow_pickle=True)
    print('The piano roll for all samples were created and stored in the directory!')


def sampling_pianoroll(piano_rolls):

    print(f"\nPlease wait! \nThe piano-rolls are being sampled...\n ")

    global group
    for track in range(len(piano_rolls)):

        class_num = piano_rolls[track][0]
        track_num = piano_rolls[track][1]
        piano_roll = piano_rolls[track][2].toarray()

        height, width = piano_roll.shape
        shift_step = 0

        if class_num in [1]:
            group = 0
            if width <= 1500:
                shift_step = 100
            if 1500 < width <= 3000:
                shift_step = 200
            if 3000 < width <= 4500:
                shift_step = 300
            if 4500 < width <= 6000:
                shift_step = 400
            if 6000 < width <= 7500:
                shift_step = 500
            if 7500 < width <= 10000:
                shift_step = 600
            if 10000 < width <= 12000:
                shift_step = 700
            if 12000 < width <= 13000:
                shift_step = 800
            if 13000 < width <= 15000:
                shift_step = 1000
            if 15000 < width:
                shift_step = 2000

        if class_num in [2, 3]:
            group = 0
            if width <= 1500:
                shift_step = 600
            if 1500 < width <= 3000:
                shift_step = 1200
            if 3000 < width <= 4500:
                shift_step = 1800
            if 4500 < width <= 6000:
                shift_step = 2400
            if 6000 < width <= 7500:
                shift_step = 3000
            if 7500 < width <= 10000:
                shift_step = 3600
            if 10000 < width <= 12000:
                shift_step = 4000
            if 12000 < width <= 13000:
                shift_step = 5000
            if 13000 < width <= 15000:
                shift_step = 5500
            if 15000 < width:
                shift_step = 6000

        if class_num in [4]:
            group = 1
            if width <= 1500:
                shift_step = 400
            if 1500 < width <= 3000:
                shift_step = 600
            if 3000 < width <= 4500:
                shift_step = 800
            if 4500 < width <= 6000:
                shift_step = 1200
            if 6000 < width <= 7500:
                shift_step = 1600
            if 7500 < width <= 10000:
                shift_step = 2000
            if 10000 < width <= 12000:
                shift_step = 2800
            if 12000 < width <= 13000:
                shift_step = 4000
            if 13000 < width <= 15000:
                shift_step = 5000
            if 15000 < width:
                shift_step = 6000

        if class_num in [5, 6]:
            group = 1
            if width <= 1500:
                shift_step = 800
            if 1500 < width <= 3000:
                shift_step = 1000
            if 3000 < width <= 4500:
                shift_step = 1200
            if 4500 < width <= 6000:
                shift_step = 1600
            if 6000 < width <= 7500:
                shift_step = 2000
            if 7500 < width <= 10000:
                shift_step = 3000
            if 10000 < width <= 12000:
                shift_step = 5000
            if 12000 < width <= 13000:
                shift_step = 6000
            if 13000 < width <= 15000:
                shift_step = 10000
            if 15000 < width:
                shift_step = 14000

        if class_num in [7, 8, 9]:
            group = 2
            if width <= 1500:
                shift_step = 800
            if 1500 < width <= 3000:
                shift_step = 1200
            if 3000 < width <= 4500:
                shift_step = 1800
            if 4500 < width <= 6000:
                shift_step = 3000
            if 6000 < width <= 7500:
                shift_step = 3600
            if 7500 < width <= 10000:
                shift_step = 5000
            if 10000 < width <= 12000:
                shift_step = 8000
            if 12000 < width <= 13000:
                shift_step = 12000
            if 13000 < width <= 15000:
                shift_step = 15000
            if 15000 < width:
                shift_step = 18000

        if class_num in [10, 11]:
            group = 3
            if width <= 1500:
                shift_step = 3000
            if 1500 < width <= 3000:
                shift_step = 4000
            if 3000 < width <= 4500:
                shift_step = 5000
            if 4500 < width <= 6000:
                shift_step = 6000
            if 6000 < width <= 7500:
                shift_step = 8000
            if 7500 < width <= 10000:
                shift_step = 10000
            if 10000 < width <= 12000:
                shift_step = 12000
            if 12000 < width <= 13000:
                shift_step = 20000
            if 13000 < width <= 15000:
                shift_step = 28000
            if 15000 < width:
                shift_step = 35000

        if class_num in [12]:
            group = 3
            if width <= 1500:
                shift_step = 10500
            if 1500 < width <= 3000:
                shift_step = 18000
            if 3000 < width <= 4500:
                shift_step = 22500
            if 4500 < width <= 6000:
                shift_step = 28000
            if 6000 < width <= 7500:
                shift_step = 34500
            if 7500 < width <= 10000:
                shift_step = 40500
            if 10000 < width <= 12000:
                shift_step = 45500
            if 12000 < width <= 13000:
                shift_step = 55000
            if 13000 < width <= 15000:
                shift_step = 60500
            if 15000 < width:
                shift_step = 70000

        start_point = 0
        sample_num = 0
        # print(Fore.GREEN + 'Pianoroll shape:', piano_roll.shape)
        # print(Fore.GREEN + 'Class: {}, Track: {}, Shift-Step-Size: {}'.format(class_num, track_num, shift_step))
        # print(Style.RESET_ALL)
        while (start_point + WINDOWS_WIDTH) < piano_roll.shape[1]:
            sample_num += 1
            png_directory = 'D:/Business/Idea_Music/Data/Original_Data/ISMLP_PngPhotos/' \
                            'pianoroll_Class{:02d}_Track{:03d}_Sample{:03d}.png'.format(group,  # [class_num -1]
                                                                                        track_num,
                                                                                        sample_num)
            # print(png_directory)
            plt.imsave(png_directory,
                       piano_roll[:, start_point: start_point + WINDOWS_WIDTH],
                       format='png')
            start_point = start_point + shift_step

        # print(Fore.BLUE + 'Number of samples:', sample_num)
        # print(Style.RESET_ALL)


if __name__ == "__main__":

    # General information of the database
    COMPLEXITY_LEVEL_NUMBER = complexity_class_number_computation(DIRECTORY)
    tracks = each_level_track_number(DIRECTORY, COMPLEXITY_LEVEL_NUMBER)

    # Converting MP3 to MIDI
    convert_mp3_to_midi(DIRECTORY)

    # Converting MIDI to PIANOROLLs
    midi_to_pianoroll()

    # Sampling the piano-roll
    database_pianoroll = np.load('database_pianoroll.npy', allow_pickle=True)
    print(database_pianoroll.shape)
    sampling_pianoroll(database_pianoroll)

    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("\nCurrent Time =", current_time)
