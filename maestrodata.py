from dataclasses import dataclass, field
import numpy as np
from typing import List
import librosa
import pandas as pd
import music21 as m21
import warnings
import torch
import math
import os
import tempfile
from midi2audio import FluidSynth

# Alright, time to wrap this up....

# First, our data is stored like this [DATA_LOCATION]/[YEAR]/file.*
# and [DATA_LOCATION]/maestro-v3.0.0.csv has all relevant meta data

@dataclass
class MaestroDataConfig:
    root_dir: str = './maestro-v3.0.0'
    meta_csv: str = 'maestro-v3.0.0.csv'
    seed: int = None
    subset: float = 1
    years: list = field(default_factory=list)
    midi_ticks_per_sec = 960
    midi_beats_per_min = 120
    midi_ticks_per_beat = midi_ticks_per_sec * 60 // midi_beats_per_min
    sec_per_sample = [ 1, 3 ]
    sr = 44100
    n_fft = 2048
    hop_length = 1024               # sr / hop_length = resolution (MFCCs per sec,) 43
    win_length = 1024               # probably best same as hop_length?
    n_mels = 128                    # number of samples per time step
    #n_mfcc = 20
    power = 2                       # 2 = power spectrum
    ticks_max_resolution = 10       # round (up) time to multiple of # of ticks
    max_ticks = 59900
    pitch_event_min = 6000
    pitch_event_max = pitch_event_min + 127
    velocity_event_min = 6000+128
    velocity_event_max = velocity_event_min + 127
    time_event_min = 10             # leaving space for padding, end of sequence
    time_event_max = 6000 -1
    #eot_event = 0
    vocab_size = 6000+128+128
    max_mfcc_size = math.ceil(5 * (44100/hop_length))    # max # of mfccs per sample
    mfcc_per_sample = tuple(sr / hop_length * np.array(sec_per_sample))
    include_velocity = False

class MaestroData:

    def __init__(self, config=MaestroDataConfig(), random_state=None):
        self.config = config
        self.random_state_init = random_state
        self.init_random_state(reset=True)
        self.df = pd.read_csv(f"{self.config.root_dir}/{self.config.meta_csv}")
        self.df = self.df.sample(frac=1, random_state=self.random_state)
        if self.config.subset != 1.0:
            self.df = self.df.groupby(['year','split'], as_index=False).apply(lambda g: g.sample((int)(self.config.subset*len(g)), random_state=self.random_state))
        if self.config.years:
            self.df = self.df[self.df.year.isin(self.config.years)]
    def init_random_state(self, reset=False):
        if reset:
            self.random_state = np.random.RandomState(seed=self.config.seed) if self.random_state_init is None else self.random_state_init
        else:
            # torch generator is properly seeded in child processes
            self.random_state = np.random.RandomState(torch.randint(0, 2**32-1, (1,)).item()) # use torch.initial_seed() ...
    def get_data(self, train=False, test=False, validation=False):
        dataset = np.array([train, test, validation]) != 0
        filter = np.array(['train', 'test', 'validation'])[dataset]
        records = self.df[self.df.split.isin(filter)]
        return records
    def load_wav(self, dr, offset_sec=0, duration_sec=None):
        #wavfile = f"{self.config.root_dir}/{dr.audio_filename}"
        wavfile = f"{dr.audio_filename}"
        return self.load_wav_from_file(wavfile, offset_sec, duration_sec)
    def load_wav_from_file(self, wavfile, offset_sec=0, duration_sec=None):
        fpath = os.path.join(self.config.root_dir, wavfile)
        y, sr = librosa.load(fpath, sr=self.config.sr, mono=True, offset=offset_sec, duration=duration_sec)
        return y, sr
    def load_mfccs(self, dr, offset_sec=0, duration_sec=None):
        #wavfile = f"{self.config.root_dir}/{dr.audio_filename}"
        #y, sr = librosa.load(wavfile, sr=self.config.sr, mono=True, offset=offset_sec, duration=duration_sec)
        y, sr = self.load_wav(dr, offset_sec, duration_sec)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length, n_mels=self.config.n_mels, power=self.config.power)
        #mel_spect = librosa.power_to_db(spect, ref=np.max) - should be included with power=2 above ...
        return mel_spect.T
    def create_midi_file_from_sample(self, dr, fname, offset_ticks, duration_ticks):
        events, remaining_ticks = self.load_midi_events(dr, offset_ticks, duration_ticks)
        self.create_midi_file(events, fname, remaining_ticks)

    def create_midi_file(self, events, fname, remaining_ticks=0):
        sf = self.create_midi(events, remaining_ticks=remaining_ticks)
        sf.open(fname, 'wb')
        sf.write()
        sf.close()

    def create_midi(self, events, remaining_ticks=0):
        events = list(events)
        events.insert(0, m21.midi.DeltaTime(track=None, time=0, channel=1))
        events.insert(1, m21.midi.MidiEvent(track=None, type=m21.midi.ChannelVoiceMessages.PROGRAM_CHANGE, channel=1))
        events[1].parameter1 = 0
        events.append(m21.midi.DeltaTime(track=None, time=remaining_ticks, channel=1))
        events.append(m21.midi.MidiEvent(track=None, type=m21.midi.MetaEvents.END_OF_TRACK))
        events[-1].data = b''

        sf = m21.midi.MidiFile()
        tr = m21.midi.MidiTrack(index=1)
        sf.format = 1
        sf.ticksPerQuarterNote = self.config.midi_ticks_per_beat
        sf.ticksPerSecond = self.config.midi_ticks_per_sec
        sf.tracks = [ tr ]
        sf.tracks[0].events = events
        sf.tracks[0].updateEvents()
        sf.tracks[0].setChannel(1)
 
        return sf
 
    def events2wav(self, events):

        midiFile = tempfile.mkstemp(suffix='.midi')[1]
        wavFile = tempfile.mkstemp(suffix='.wav')[1]
        try:
            mf = self.create_midi_file(maestro_data.unmap_events(events), fname=midiFile)
            FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2").midi_to_audio(midiFile, wavFile)
            y, sr = librosa.load(wavFile, sr=self.config.sr, mono=True)
            return y,sr
        finally:
            os.remove(wavFile)
            os.remove(midiFile)

    def load_midi_events(self, dr, offset_ticks=0, duration_ticks=None):
        midifile = f"{self.config.root_dir}/{dr.midi_filename}"
        return self.load_midi_events_from_file(midifile, offset_ticks, duration_ticks)

    def load_midi_events_from_file(self, midifile, offset_ticks=0, duration_ticks=None):
        mf = m21.midi.MidiFile()
        mf.open(midifile)
        mf.read()
        mf.close()

        ticks_end = None if duration_ticks is None else int(offset_ticks + duration_ticks)
        ticks_curr, ticks_last_event = 0, int(offset_ticks)
        events = []

        for e in mf.tracks[1].events if len(mf.tracks) > 1 else mf.tracks[0].events:
            if e.isDeltaTime():
                ticks_curr = ticks_curr + e.time
                if not ticks_end is None and ticks_curr > ticks_end:
                    break
            elif not e.time is None and e.time > 0:
                warnings.warn(f"unexpected time {e.time}>0 for non-time event, {e}, p1={e.parameter1}, p2={e.parameter2}")
            if ticks_curr >= offset_ticks:
                if e.type == m21.midi.ChannelVoiceMessages.NOTE_ON:
                    # insert delta time
                    #dt = ticks_curr-ticks_last_event
                    #if dt > self.config.time_event_max:
                    #    warnings.warn(f"delta time too long, {dt} > {self.config.time_event_max}. inserting repeated DTs.")
                    #while dt > 0:
                    #    event_dt = min(dt, self.config.time_event_max)
                    #    ne = m21.midi.DeltaTime(track=None, time=event_dt, channel=1)
                    #    events.append(ne)
                    #    dt = dt - event_dt
                    ne = m21.midi.DeltaTime(track=None, time=ticks_curr-ticks_last_event, channel=1)
                    events.append(ne)
                    # insert note on
                    ne = m21.midi.MidiEvent(track=None, type=m21.midi.ChannelVoiceMessages.NOTE_ON, time=0, channel=1)
                    ne.pitch = e.pitch
                    ne.velocity = e.velocity
                    ne.parameter1 = e.parameter1
                    ne.parameter2 = e.parameter2
                    events.append(ne)
                    ticks_last_event = ticks_curr
                elif e.isDeltaTime():
                    pass
                elif e.type == m21.midi.ChannelVoiceMessages.CONTROLLER_CHANGE and e.parameter1 in (64,66,67):
                    # pedals
                    pass
                elif e.type == m21.midi.ChannelVoiceMessages.PROGRAM_CHANGE and e.parameter1 in (0,):
                    # pedals
                    pass
                elif e.type == m21.midi.MetaEvents.END_OF_TRACK:
                    pass
                else:
                    # assumptions ...
                    warnings.warn(f"unexpected event {e.type}, p1={e.parameter1}, p2={e.parameter2}, {e}")
                    
        return events, 0 if ticks_end is None else max(ticks_end - ticks_last_event, 0)

    def unmap_events(self, events):
        config = self.config
        # abs-time, pitch, velocity, duration
        notes = np.zeros((len(events), 4), dtype=np.int)
        note_count = 0
        curr_time = 0
        for ndx, e in enumerate(events):
            if e >= config.time_event_min and e <= config.time_event_max:
                time = (e - config.time_event_min) * config.ticks_max_resolution
                curr_time = max(time, curr_time)
                prior_event_is_pitch = False
            elif e >= config.pitch_event_min and e <= config.pitch_event_max:
                pitch = e - config.pitch_event_min
                notes[note_count] = [ curr_time, pitch, -1, -1 ]
                note_count = note_count+1
            elif e >= config.velocity_event_min and e <= config.velocity_event_max:
                velocity = e - config.velocity_event_min
                if note_count > 0 and notes[note_count-1, 3] == -1:
                    notes[note_count-1, 3] = velocity
            else:
                # bad event
                pass
        notes = notes[:note_count, :]
        # default value for velocity and duration
        notes[notes[:, 2]<0, 2] = 64
        notes[notes[:, 3]<0, 3] = config.midi_ticks_per_sec / 4
        # create on/off events & sort
        notes_off = np.concatenate([notes[:, [0]]+notes[:, [3]], notes[:, [1]], np.zeros_like(notes[:, :2])], axis=1)
        notes = np.concatenate([notes, notes_off], axis=0)
        notes = notes[notes[:, 0].argsort(), :]
        # add prior time
        pt = np.insert(notes[:-1, 0], 0, 0) if note_count>0 else notes[:-1, 0]
        notes = np.c_[notes, pt]
        # FORMAT: abs-time, pitch, velocity, duration, prior-time
        def mapf(row):
            m1 = m21.midi.DeltaTime(track=None, time=0, channel=1)
            m1.time = int(row[0] - row[4])
            m2 = m21.midi.MidiEvent(track=None, type=m21.midi.ChannelVoiceMessages.NOTE_ON, time=0, channel=1)
            m2.velocity = int(row[2])
            m2.pitch = int(row[1])
            #m2.parameter1 = int(row[1])
            #m2.parameter2 = row[2]
            #m2.time = row[0] - row[4]
            return [m1, m2]
        # map to midi events
        return np.apply_along_axis(mapf, 1, notes).reshape(-1).tolist() if note_count>0 else []

    def map_midi_events(self, midi, remaining_ticks=0, load_all=False):
        assert len(midi)==0 or midi[0].isDeltaTime(), f"error, sequence must start with deltatime but is {midi[0]}"
        config = self.config

        def midi2ev(ev):
            if ev.isDeltaTime():
                return ['DT', ev.time, -1, -1, '']
            elif ev.type == m21.midi.ChannelVoiceMessages.NOTE_ON:
                if ev.velocity > 0:
                    return ['NT', 0, ev.pitch, ev.velocity, '']
                else:
                    # skipping these for now ...
                    return ['IG', 0, ev.pitch, ev.velocity, 'note off']
            elif ev.type == m21.midi.ChannelVoiceMessages.CONTROLLER_CHANGE and e.parameter1 in (64,66,67):
                return ['IG', 0, 0, 0, 'pedals']
            elif e.type == m21.midi.ChannelVoiceMessages.PROGRAM_CHANGE and e.parameter1 in (0,):
                return ['IG', 0, 0, 0, 'prog change']
            elif e.type == m21.midi.MetaEvents.END_OF_TRACK:
                return ['IG', 0, 0, 0, 'end of track']
            else:
                return ['WA', 0, 0, 0, f"unexpected event {e.type}, p1={e.parameter1}, p2={e.parameter2}, {e}"]

        events = [ midi2ev(_) for _ in midi ]
        if remaining_ticks > 0:
            events.append(['DT', remaining_ticks, -1, -1, '' ])
        events = pd.DataFrame(events, columns=['type','rel_ticks','pitch','velocity','message'])

        # warnings
        for _ in events[events.type=='WA'].iterrows():
            warnings.warn(_.message)
        events = events[events.type.isin(['DT','NT'])]
        bad = ((events.type=='NT') & ((events.pitch<0) | (events.pitch>=128)))
        if bad.any():
            warnings.warn(f"there are {len(bad)} records with invalid pitch")
            events = events.loc[~bad]
        bad = ((events.type=='NT') & ((events.velocity<0) | (events.velocity>=128)))
        if bad.any():
            warnings.warn(f"there are {len(bad)} records with invalid velocity")
            events = events.loc[~bad]

        # calculate absolute times
        events.loc[:, 'abs_ticks'] = events['rel_ticks'].cumsum()
        if not load_all and len(events)>0 and events.iloc[-1].abs_ticks > config.max_ticks:
            warnings.warn(f"segment too long, ticks {events.iloc[-1].abs_ticks} > {config.max_ticks}")
            start = (events.iloc[-1].abs_ticks > config.max_ticks).idmax()
            events = events.iloc[:start]
        events.loc[:, 'abs_time'] = (events['abs_ticks']/config.ticks_max_resolution).round().astype(int)
        events.loc[:, 'event_id'] = -1

        # calculate event IDs
        e = events.loc[events.type=='DT']
        events.loc[e.index, 'event_id'] = (config.time_event_min + e.abs_time).clip(None, config.time_event_max)

        e = events.loc[events.type=='NT']
        events.loc[e.index, 'event_id'] = (config.pitch_event_min + e.pitch).clip(None, config.pitch_event_max)

        # remove duplicate time events
        events['prev_abs_time'] = events.abs_time.shift(1)
        dups = (events.type=='DT') & (events.abs_time == events.prev_abs_time)
        events = events.loc[~dups, :]
        return events

    def sample_record(self, dr):
        duration_sec_range = np.array([self.config.sec_per_sample[0], min(self.config.sec_per_sample[1], dr.duration)])
        duration_sec = self.random_state.rand() * (duration_sec_range[1]-duration_sec_range[0]) + duration_sec_range[0]
        offset_sec = self.random_state.rand() * (dr.duration - duration_sec)
        return offset_sec, duration_sec

    def load_sample(self, dr, offset_sec=0, duration_sec=None, load_mfcc=True, load_midi=True):
        mfccs = self.load_mfccs(dr, offset_sec, duration_sec) if load_mfcc else None
        if load_midi:
            midi_events = self.load_midi_events(dr, offset_sec*self.config.midi_ticks_per_sec,
                None if duration_sec is None else duration_sec*self.config.midi_ticks_per_sec)
            events = self.map_midi_events(*midi_events, load_all=(duration_sec is None))
        else:
            events = None
            midi_events = None
        return mfccs, events, midi_events, (dr.midi_filename, dr.audio_filename, offset_sec, duration_sec)

    # returns the max sequence length (event count) given max sequence length (in seconds), excluding BOS and EOS
    def max_record_seq_length(self, events, sequence_len_sec=None):
        if sequence_len_sec is None:
            sequence_len_sec = self.config.sec_per_sample[1]
        start_ndx, stop_ndx = 0, 0
        dts = events[events.type=='DT'][['time_abs_sec']]
        max_len = 0
        for index, row in dts.iterrows():
            stop_ndx = index
            start_ndx = (dts.loc[start_ndx:stop_ndx] >= row[0]-sequence_len_sec).idxmax()[0]
            max_len = max(max_len, stop_ndx-start_ndx+1)
        return max_len

    def max_seq_length(self, sequence_len_sec=None, tqdm=None):
        max_len = 0
        for _, row in self.df.iterrows() if tqdm is None else tqdm(self.df.iterrows(), total=len(self.df)):
            rec = self.load_sample(row, load_mfcc=False)
            max_len = max(self.max_record_seq_length(rec[1]), max_len)
        return max_len

@dataclass
class MaestroDatasetConfig(MaestroDataConfig):

    def __init__(self):
        self.max_source_length = self.max_mfcc_size

    target_padding_id : int = 0
    target_bos_id : int = 0.  # leave this 0 to match with T5's decoder_start_token_id
    target_eos_id : int = 1  # leave this 1 or ? -100 to match with T5's e
    mask_id : int = 1
    mask_padding_id : int = 0
    label_padding_id : int = -100
    max_source_length : int = 216  # enough for 5 seconds of 44.1kHz with hop_length 1024
    max_target_length : int = 1024 # guess ... for 2017 data, 984 is max length excl. BOS & EOS
    mfcc_pad_value : int = 0

class MaestroDataset(torch.utils.data.Dataset):

    def __init__(self, data, batch_size=5, max_size=None, train=False, test=False, validation=False, fixed_sample=False, include_meta_data=False):
        self.config = data.config
        self.batch_size = batch_size
        self.data = data
        self.max_size = max_size
        self.records = data.get_data(train=train, test=test, validation=validation)
        self.fixed_sample = fixed_sample
        self.include_meta_data = include_meta_data
        if not self.max_size is None and len(self.records) > self.max_size:
            self.records = self.records.iloc[:max_size]
        if self.fixed_sample:
            warnings.warn(f"Using {len(self.records)} fixed sample(s)")
            self.samples = []

    def init_random_state(self):
        self.data.init_random_state(reset=self.fixed_sample)

    def worker_init_fn(self, worker_id):
        self.init_random_state()

    def pad(self, tokens):
        if len(tokens) >= self.config.max_target_length-2:
            warnings.warn(f"target token length {len(tokens)} > {self.config.max_target_length}-3 too long.")
            tokens = tokens[:(self.config.max_target_length-3)]
        #tokens = np.append(tokens, [self.config.target_eos_id], axis=0)
        #pad_length = self.config.max_target_length+1 - len(tokens)
        target = np.pad(tokens, (1,self.config.max_target_length+1-tokens.shape[0]-1), mode='constant', constant_values=(self.config.target_bos_id, self.config.target_padding_id))
        #
        #maybe this is not required ??? (nov20)
        #
        #target[tokens.shape[0]+1] = self.config.target_eos_id
        target[tokens.shape[0]+1] = self.config.target_eos_id
        labels = np.pad(tokens, (0,self.config.max_target_length+1-tokens.shape[0]), mode='constant', constant_values=self.config.label_padding_id)
        labels[tokens.shape[0]]  = self.config.target_eos_id
        mask = np.full_like(labels, self.config.mask_padding_id)
        #
        #maybe this is not required for EOS ??? (nov20)
        #
        mask[:len(tokens)+1] = self.config.mask_id
        r = {'target_id':target, 'label':labels, 'attention_mask':mask}
        return r

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        start = idx * self.batch_size
        if self.fixed_sample:
            self.samples = self.samples + [ self.data.sample_record(self.records.iloc[_]) for _ in range(len(self.samples), idx+1) ]
            sample = self.samples[idx]
        else:
            sample = self.data.sample_record(self.records.iloc[idx])
        end = min((idx+1) * self.batch_size, len(self.records))
        x = [ self.data.load_sample(self.records.iloc[idx], *sample) for idx in range(start, end) ]
        mfcc = [ _[0] for _ in x ]
        mfcc = [ np.pad(_, ((0,(self.config.max_source_length-_.shape[0])),(0,0)), mode='constant', constant_values=self.config.mfcc_pad_value) for _ in mfcc ]
        midi = [ self.pad( _[1].event_id ) for _ in x ]
        if self.include_meta_data:
            actual_midi = [ _[2] for _ in x ]
            sample = [ _[3] for _ in x ]
        if end == start + 1:
            data = {
                'mfcc' : mfcc[0],
                'target_id' : midi[0]['target_id'],
                'label' : midi[0]['label'],
                'attention_mask' : midi[0]['attention_mask'],
                }
        else:
            data = {
                'mfcc' : np.stack(mfcc),
                'target_id' : np.stack([ _['target_id'] for _ in midi ]),
                'label' : np.stack([ _['label'] for _ in midi ]),
                'attention_mask' : np.stack([ _['attention_mask'] for _ in midi ]),
            }
        if self.include_meta_data:
            data['midi'] = actual_midi
            data['sample'] = sample
        data['mfcc'] = torch.tensor(data['mfcc'], dtype=torch.float32)
        data['target_id'] = torch.tensor(data['target_id'], dtype=torch.int64)
        data['label'] = torch.tensor(data['label'], dtype=torch.int64)
        data['attention_mask'] = torch.tensor(data['attention_mask'], dtype=torch.int64)
        return data

    def __len__(self):
        return (len(self.records)+self.batch_size-1)//self.batch_size

