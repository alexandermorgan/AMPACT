import pandas as pd
import numpy as np
import music21 as m21
import math
import pdb

# adjustable constants
score_path = './test_files/polyphonic4voices1note.mei'
score_path = './test_files/B070_00_03c_b.krn'
score_path = './test_files/busnoys.krn'     
# # TODO: look into why busnoy import raises an error
#  - krn annotations associated with score times same dims as mask
#  - krn import 
#  - **harm and **function importing 
# score_path = './test_files/M025_00_01a_a-repeated.krn'
aFreq = 440
bpm = 60
winms = 100
sample_rate = 2000
num_harmonics = 1
width = 0
base_note = 0
tuning_factor = 1
fftlen = 2**round(math.log(winms / 1000 * sample_rate) / math.log(2))

# conversion of score to music21 format
class Score(score_path):
  def __init__(self, score, path, mei_doc=None, date=None):
    self.path = score_path
    self.score = m21.converter.parse(score_path)
    self.file_name = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    self.file_extension = score_path.rsplit('.', 1)[1]
    self.metadata = {'Title': score.metadata.title, 'Composer': score.metadata.composer}
    self.part_streams = self.score.getElementsByClass(m21.stream.Part)
    self.semi_flat_parts = [part.semiFlat for part in self.part_streams]
    self.part_names = []
    self.analyses = {}
    for i, part in enumerate(self.semi_flat_parts):
      name = part.partName if (part.partName and part.partName not in self.part_names) else 'Part-' + str(i + 1)
      self.part_names.append(name)
    self.parts = []
    for i, flat_part in enumerate(self.semi_flat_parts):
      elements = flat_part.getElementsByClass(['Note', 'Rest', 'Chord'])
      events, offsets = [], []
      for nrc in elements:
        if nrc.isChord:
          events.append(nrc.notes)
          offsets.append(round(float(nrc.offset), 4))
        else:
          events.append((nrc,))
          offsets.append(round(float(nrc.offset), 4))
      df = pd.DataFrame(events, index=offsets)
      df = df.add_prefix(self.part_names[i] + '_')
      # for now remove multiple events at the same offset in a given part
      df = df[~df.index.duplicated(keep='last')]
      self.parts.append(df)
    self.m21_objects = pd.concat(self.parts, names=self.part_names, axis=1)

  def lyrics(self):
    if 'lyrics' not in self.analysis:
      self.analyses['lyrics'] = self.m21_objects.applymap(lambda cell: cell.lyric or np.nan, na_action='ignore').dropna(how='all')
    return self.analyses['lyrics']

  def _remove_tied(noteOrRest):
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
      return pd.NA
    return noteOrRest

  def m21ObjectsNoTies(self):
    if 'm21ObjectsNoTies' not in self.analyses:
      self.analyses['m21ObjectsNoTies'] = self.m21_objects.applymap(self._remove_tied).dropna(how='all')
    return self.analyses['m21ObjectsNoTies']

  def midi_pitches(self):
    '''\tProcess notes as midi pitches. Midi does not have a representation
    for rests, so use -1 as a placeholder.'''
    if 'midi_pitches' not in self.analyses:
      midi_pitches = self.m21ObjectsNoTies().applymap(lambda noteOrRest: -1 if noteOrRest.isRest else noteOrRest.pitch.midi, na_action='ignore')
      midi_pitches = midi_pitches.ffill().astype(int)
      if not all([rest == -1 for rest in midi_pitches.iloc[-1, :]]):
        midi_pitches.loc[score.highestTime, :] = -1   # add "rests" to end if the last row is not already all rests
      self.analyses['midi_pitches'] = midi_pitches
    return self.analyses['midi_pitches']

  def piano_roll(self):
    '''\tConstruct midi piano roll. NB: there are 128 possible midi pitches.'''
    if 'piano_roll' not in self.analyses:
      piano_roll = pd.DataFrame(index=range(128), columns=midi_pitches.index.values)
      for offset in midi_pitches.index:
        for pitch in midi_pitches.loc[offset]:
          if pitch >= 0:
            piano_roll.at[pitch, offset] = 1
      piano_roll.fillna(0, inplace=True)
      self.analyses['piano_roll'] = piano_roll
    return self.analyses['piano_roll']

  def sampled(self, winms=100, sample_rate=2000, bpm=60)
    '''\tSample the score according to bpm, sample_rate, and winms.'''
    key = ('sampled', winms, sample_rate, bpm)
    if key not in self.analyses:
      # freqs = [round(2**((i-69)/12) * aFreq, 3) for i in range(128)] 
      # piano_roll.index = freqs
      pr_cols = self.piano_roll().columns
      col_set = set(pr_cols)  # this set makes sure that any timepoints in piano_roll cols will persist
      slices = 60/bpm * 20
      col_set.update([t/slices for t in range(0, int(pr_cols[-1] * slices) + 1)])
      num_rows = int(2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
      sampled = pd.DataFrame(columns=sorted(col_set), index=range(num_rows))
      sampled.update(self.piano_roll())
      sampled = sampled.ffill(axis=1).fillna(0).astype(int)
      self.analyses[key] = sampled
    return self.analyses[key]

  def mask(self, winms=100, sample_rate=2000, num_harmonics=1, width=0,
            bpm=60, aFreq=440, base_note=0, tuning_factor=1)
    '''\tConstruct a mask from the sampled piano roll using width and harmonics.'''
    key = ('mask', winms, sample_rate, num_harmonics, width, bpm, aFreq, base_note, tuning_factor)
    if key not in self.analyses:
      width_semitone_factor = 2 ** ((width / 2) / 12)
      sampled = self.sampled(winms, sample_rate, bpm)
      noprows = sampled.shape[0]
      mask = sampled * 0
      fftlen = 2**round(math.log(winms / 1000 * sample_rate) / math.log(2))

      for row in range(base_note, sampled.shape[0]):
        note = base_note + row
        # MIDI note to Hz: MIDI 69 = 440 Hz = A4
        freq = tuning_factor * (2 ** (note / 12)) * aFreq / (2 ** (69 / 12))
        if sampled.loc[row, :].sum() > 0:
          mcol = pd.Series(0, index=range(noprows))
          for harm in range(1, num_harmonics + 1):
            minbin = math.floor(harm * freq / width_semitone_factor / sample_rate * fftlen)
            maxbin = math.ceil(harm * freq * width_semitone_factor / sample_rate * fftlen)
            if minbin <= noprows:
              maxbin = min(maxbin, noprows)
              mcol.loc[minbin : maxbin] = 1
          mask.iloc[np.where(mcol)[0], np.where(sampled.iloc[row])[0]] = 1
      self.analyses[key] = mask
    return self.analyses[key]


  # debugging print statements
  # m2 = mask[mask.sum(axis=1) > 0]
  # ser = m2.index.to_series()
  # ends = m2[(ser != (ser.shift() + 1)) | (ser != (ser.shift(-1) -1))]
  # sums = ends.sum()
  # corners = ends.loc[:, ((sums != sums.shift()) | (sums != sums.shift(-1)))]
  # print(corners)
  # vert = ends.T
  # print('shape ->', mask.shape)
  # pdb.set_trace()
