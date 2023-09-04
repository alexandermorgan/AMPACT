import pandas as pd
import numpy as np
import music21 as m21
import math
import pdb

# adjustable constants
score_path = './test_files/polyExample2voices1note.mid'
aFreq = 440
bpm = 60
winms = 100
sample_rate = 4000
num_harmonics = 1
width = 1
base_note = 0
tuning_factor = 1
fftlen = 2**round(math.log(winms / 1000 * sample_rate) / math.log(2))

# basic indexing of score
score = m21.converter.parse(score_path)
parts = score.getElementsByClass(m21.stream.Part)
semi_flat_parts = [part.semiFlat for part in parts]
part_names = []
for i, part in enumerate(semi_flat_parts):
  name = part.partName or 'Part-' + str(i + 1)
  if name in part_names:
    name = 'Part-' + str(i + 1)
  elif '_' in name:
    print('\n*** Warning: it is problematic to have an underscore in a part name so _ was replaced with -. ***\n')
    name = name.replace('_', '-')
  part_names.append(name)
part_series = []
for i, flat_part in enumerate(semi_flat_parts):
  notesAndRests = flat_part.getElementsByClass(['Note', 'Rest', 'Chord'])
  notesAndRests = [max(noteOrRest.notes) if noteOrRest.isChord else noteOrRest for noteOrRest in notesAndRests]
  ser = pd.Series(notesAndRests, name=part_names[i])
  ser.index = ser.apply(lambda noteOrRest: round(float(noteOrRest.offset), 4))
  # for now remove multiple events at the same offset in a given part
  ser = ser[~ser.index.duplicated()]
  part_series.append(ser)
m21_objects = pd.concat(part_series, names=part_names, axis=1)
def _remove_tied(noteOrRest):
  if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
    return pd.NA
  return noteOrRest
m21ObjectsNoTies = m21_objects.applymap(_remove_tied).dropna(how='all')

# process notes as midi pitches
def _midiPitchHelper(noteOrRest):
  """midi does not have a representation for rests, so use -1 as a placeholder."""
  if noteOrRest.isRest:
    return -1
  return noteOrRest.pitch.midi
midi_pitches = m21ObjectsNoTies.applymap(_midiPitchHelper, na_action='ignore')
midi_pitches = midi_pitches.ffill().astype(int)
if not all([rest == -1 for rest in midi_pitches.iloc[-1, :]]):
    midi_pitches.loc[score.highestTime, :] = -1   # add "rests" to end if the last row is not already all rests

# construct midi piano roll / mask, NB: there are 128 possible midi pitches
_piano_roll = pd.DataFrame(index=range(1, 129), columns=midi_pitches.index.values)
def _reshape(row):
  for midiNum in row.values:
    if midiNum > -1:
      _piano_roll.at[midiNum, row.name] = 1
midi_pitches.apply(_reshape, axis=1)
_piano_roll.fillna(0, inplace=True)

for h, col in enumerate(midi_pitches.columns):
  part = midi_pitches.loc[:, col].dropna()
  partIndexInPianoRoll = part.index.to_series().apply(lambda i: _piano_roll.columns.get_loc(i))
  for i, row in enumerate(part.index[:-1]):
    pitch = int(part.at[row])
    start = partIndexInPianoRoll[row]
    end = partIndexInPianoRoll[part.index[i + 1]]
    if pitch > -1:
      _piano_roll.loc[pitch, start:end] = 1
    else: # current event is a rest
      if i == 0:
        continue
      pitch = int(part.at[i - 1])
      if pitch == -1:
        continue
      if _piano_roll.at[pitch, start] != 1: # don't overwrite a note with a rest
        _piano_roll.at[pitch, start] = 0

piano_roll = _piano_roll.ffill(axis=1).fillna(0).astype(int)

# sample the score according to bpm, sample_rate, and winms
# freqs = [round(2**((i-69)/12) * aFreq, 3) for i in range(128)] 
# piano_roll.index = freqs
col_set = set(piano_roll.columns)  # this set makes sure that any timepoints in piano_roll cols will persist
slices = 60/bpm * 20
col_set.update([t/slices for t in range(0, int(piano_roll.columns[-1] * slices) + 1)])
num_rows = int(2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
sampled = pd.DataFrame(columns=sorted(col_set), index=range(num_rows))
sampled.update(piano_roll)
sampled = sampled.ffill(axis=1).fillna(0).astype(int)

# construct a mask that applies width and harmonics
width_semitone_factor = 2 ** ((width / 2) / 12)
noprows = sampled.shape[0]
mask = sampled * 0

for row in range(base_note, sampled.shape[0]):
  note = base_note + row
  # MIDI note to Hz: MIDI 69 = 440 Hz
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


# debugging print statements
print({'winms': winms, 'sample_rate': sample_rate, 'num_harmonics':num_harmonics, 'width': width, 'piece': score_path})
m2 = mask[mask.sum(axis=1) > 0]
ser = m2.index.to_series()
ends = m2[(ser != (ser.shift() + 1)) | (ser != (ser.shift(-1) -1))]
print(ends)
vert = ends.T
print('shape ->', mask.shape)
pdb.set_trace()
