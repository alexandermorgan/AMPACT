import pandas as pd
import music21 as m21
import numpy as np
import pdb


score_path = '/Users/amor/Desktop/Code/CUNY/exampleOneNote.mid'
score = m21.converter.parse(score_path)
_memos = {}
aFreq = 440
width = 0

_memos['parts'] = score.getElementsByClass(m21.stream.Part)

_memos['semiFlatParts'] = [part.semiFlat for part in _memos['parts']]

part_names = []
for i, part in enumerate(_memos['semiFlatParts']):
  name = part.partName or 'Part-' + str(i + 1)
  if name in part_names:
    name = 'Part-' + str(i + 1)
  elif '_' in name:
    print('\n*** Warning: it is problematic to have an underscore in a part name so _ was replaced with -. ***\n')
    name = name.replace('_', '-')
  part_names.append(name)
_memos['partNames'] = part_names

part_series = []
for i, flat_part in enumerate(_memos['semiFlatParts']):
  notesAndRests = flat_part.getElementsByClass(['Note', 'Rest', 'Chord'])
  notesAndRests = [max(noteOrRest.notes) if noteOrRest.isChord else noteOrRest for noteOrRest in notesAndRests]
  ser = pd.Series(notesAndRests, name=part_names[i])
  ser.index = ser.apply(lambda noteOrRest: noteOrRest.offset)
  # for now remove multiple events at the same offset in a given part
  ser = ser[~ser.index.duplicated()]
  part_series.append(ser)
_memos['partSeries'] = part_series

_memos['m21Objects'] = pd.concat(_memos['partSeries'], names=_memos['partNames'], axis=1)

def _remove_tied(noteOrRest):
  if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
    return np.nan
  return noteOrRest
_memos['m21ObjectsNoTies'] = _memos['m21Objects'].applymap(_remove_tied).dropna(how='all')

def _midiPitchHelper(noteOrRest):
  """midi does not have a representation for rests, so use -1 as a placeholder."""
  if noteOrRest.isRest:
    return -1
  return noteOrRest.pitch.midi
_memos['midiPitches'] = _memos['m21ObjectsNoTies'].applymap(_midiPitchHelper, na_action='ignore')
_memos['midiPitches'] = _memos['midiPitches'].ffill().astype(int)
# add "rests" to end to effectively give correct duration to last note in each voice
_memos['midiPitches'].loc[score.highestTime, :] = -1
mp = _memos['midiPitches']

_possibleMidiPitches = list(range(128))
_pianoRoll = pd.DataFrame(index=_possibleMidiPitches, columns=_memos['midiPitches'].index.values)

def _reshape(row):
  for midiNum in row.values:
    if midiNum > -1:
      _pianoRoll.at[midiNum, row.name] = 1
_memos['midiPitches'].apply(_reshape, axis=1)
_pianoRoll.fillna(0, inplace=True)

for h, col in enumerate(_memos['midiPitches'].columns):
  part = _memos['midiPitches'].loc[:, col].dropna()
  partIndexInPianoRoll = part.index.to_series().apply(lambda i: _pianoRoll.index.get_loc(i))
  for i, row in enumerate(part.index[:-1]):
    pitch = int(part.at[row])
    start = partIndexInPianoRoll[row]
    end = partIndexInPianoRoll[part.index[i + 1]]
    if pitch > -1:
      _pianoRoll.iloc[pitch, start:end] = 1
    else: # current event is a rest
      if i == 0:
        continue
      pitch = int(part.iat[i - 1])
      if pitch == -1:
        continue
      if _pianoRoll.iat[pitch, start] != 1: # don't overwrite a note with a rest
        _pianoRoll.iat[pitch, start] = 0

_memos['pianoRoll'] = _pianoRoll.ffill(axis=1).fillna(0).astype(int)
freqs = [round(2**((i-69)/12) * aFreq, 3) for i in range(128)]
_memos['pianoRoll'].index = freqs
sampled = pd.DataFrame(columns=[t/20  for t in range(0, int(score.highestTime) * 20 + 1)], index=freqs)
sampled.update(_memos['pianoRoll'])
_memos['pianoRoll'] = sampled.ffill(axis=1)
pr = _memos['pianoRoll'].copy()
if width > 0:
  pr = pr.replace(0, pd.NA).ffill(limit=width).bfill(limit=width).fillna(0)
print(pr.iloc[65:75, :21])
pdb.set_trace()
# _memos['pianoRoll'].to_csv('path_to_csv_file.csv')
