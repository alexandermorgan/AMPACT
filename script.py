import pandas as pd
import music21 as m21
import math
import pdb

# adjustable constants
score_path = './test_files/monophonic1note.mid'
aFreq = 440
width = 0
bpm = 60
winms = 200
sample_rate = 4000
base_note = 1
tuning_factor = 1
num_harmonics = 1

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
_piano_roll = pd.DataFrame(index=range(128), columns=midi_pitches.index.values)
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
      _piano_roll.iloc[pitch, start:end] = 1
    else: # current event is a rest
      if i == 0:
        continue
      pitch = int(part.iat[i - 1])
      if pitch == -1:
        continue
      if _piano_roll.iat[pitch, start] != 1: # don't overwrite a note with a rest
        _piano_roll.iat[pitch, start] = 0

piano_roll = _piano_roll.ffill(axis=1).fillna(0).astype(int)
# freqs = [round(2**((i-69)/12) * aFreq, 3) for i in range(128)] 
# piano_roll.index = freqs
col_set = set(piano_roll.columns)  # this set makes sure that any timepoints in piano_roll cols will persist
slices = 60/bpm * 20
col_set.update([t/slices for t in range(0, int(piano_roll.columns[-1] * slices) + 1)])
num_rows = int(2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
sampled = pd.DataFrame(columns=sorted(col_set), index=range(num_rows))
sampled.update(piano_roll)
sampled = sampled.ffill(axis=1).fillna(0).astype(int)

width_semitone_factor = 2**(width/6)  # use 6 instead of 12 since it is applied above and below
mask = sampled * 0
for row in range(base_note - 1, base_note + 127):    # there are 128 distinct midi_pitches 
  note = base_note - 1 + row
  freq = tuning_factor * 2**(note/12) * aFreq / 2**(69/12)
  if sum(sampled.iloc[row]) > 0:
    row_mask = sampled.loc[row] > 0
    for harmonic in range(2, num_harmonics + 2):
      min_freq = 1 + math.floor(harmonic * freq / width_semitone_factor / sample_rate * num_rows)
      max_freq = 1 + math.ceil(harmonic * freq * width_semitone_factor / sample_rate * num_rows)
      print({'note':note, 'freq':freq, 'harmonic':harmonic, 'min':min_freq, 'max':max_freq})
      if min_freq <= num_rows:
        max_freq = min(max_freq, num_rows)
        mask.loc[min_freq:max_freq, row_mask] = 1


print(mask.loc[100:150, :])
m2 = mask.replace(0, pd.NA).dropna(how='all')
print(m2)
pdb.set_trace()

if width > 0:
  sampled = sampled.replace(0, pd.NA).ffill(limit=width).bfill(limit=width).fillna(0)
test = sampled.iloc[125:135, 60:90].copy()
print(sampled)
pdb.set_trace()
# piano_roll.to_csv('path_to_csv_file.csv')
