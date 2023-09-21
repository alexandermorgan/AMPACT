import pandas as pd
import numpy as np
import music21 as m21
# from m21.humdrum.spineParser.HumdrumFile
import math
import pdb

# score_path = './test_files/polyphonic4voices1note.mei'
# score_path = './test_files/B070_00_03c_b.krn'
# score_path = './test_files/mozart.krn'
imported_scores = {}

class Score:
  '''\tImport score via music21 and expose AMPACT's analysis utilities which are
  generally formatted as Pandas DataFrames.'''
  def __init__(self, score_path):
    self.path = score_path
    if score_path not in imported_scores:
      imported_scores[score_path] = m21.converter.parse(score_path)
    self.score = m21.converter.parse(score_path)
    self.file_name = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    self.file_extension = score_path.rsplit('.', 1)[1]
    self.metadata = {'Title': self.score.metadata.title, 'Composer': self.score.metadata.composer}
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
  
  def _import_function_harm_spines(self):
    if self.file_extension == 'krn':
      humFile = m21.humdrum.spineParser.HumdrumFile(self.path)
      humFile.parseFilename()
      objs = self.m21_objects()
      for spine in humFile.spineCollection:
        if spine.spineType in ('harm', 'function'):
          vals, valPositions = [], []
          keys, keyPositions = [], []
          start = False
          for event in spine.eventList:
            contents = event.contents
            if contents.endswith(':') and contents.startswith('*'):
              start = True
              keys.append(contents)
              keyPositions.append(event.position)
              continue
            elif not start or '!' in contents or '=' in  contents or '*-' == contents:
              continue
            elif start:
              vals.append(contents)
              valPositions.append(event.position)

          df1 = self._priority()
          name = spine.spineType.title()
          df2 = pd.DataFrame({name: vals}, index=valPositions)
          joined = df1.join(df2, on='Priority')
          self.analyses[spine.spineType] = pd.Series(joined[name].values, index=joined.Offset)

      if 'functions' not in self.analyses:
        self.analyses['functions'] = pd.Series()
      if 'harmonies' not in self.analyses:
        self.analyses['harmonies'] = pd.Series()
      if 'keys' not in self.analyses:
        self.analyses['keys'] = pd.Series()
  
  def m21_objects(self):
    if 'm21_objects' not in self.analyses:
      self.analyses['m21_objects'] = pd.concat(self.parts, axis=1, sort=True)
    return self.analyses['m21_objects']

  def lyrics(self):
    if 'lyrics' not in self.analyses:
      self.analyses['lyrics'] = self.m21_objects().applymap(lambda cell: cell.lyric or np.nan, na_action='ignore').dropna(how='all')
    return self.analyses['lyrics']

  def _priority(self):
    '''\tFor .krn files, get the line numbers of the events in the piece, which music21
    often calls "priority". For other encoding formats return an empty dataframe.'''
    if '_priority' in self.analyses:
      return self.analyses['_priority']
    if self.file_extension != 'krn':
      priority = pd.DataFrame()
    else:
      priority = self.m21_objects().applymap(lambda cell: cell.priority, na_action='ignore').ffill(axis=1).iloc[:, -1].astype(int)
      priority = pd.DataFrame({'Priority': priority.values, 'Offset': priority.index})
    self.analyses['_priority'] = priority
    return priority

  def harmonies(self):
    '''\tGet the harmonic analysis from the **harm spine as a pandas series if this
    piece is a kern file and has a **harm spine. Otherwise return an empty series.'''
    if 'harm' not in self.analyses:
      self._import_function_harm_spines()
    return self.analyses['harm']

  def functions(self):
    '''\tGet the functional analysis from the **function spine as a pandas series if this
    piece is a kern file and has a **function spine. Otherwise return an empty series.'''
    if 'function' not in self.analyses:
      self._import_function_harm_spines()
    return self.analyses['function']

  def _remove_tied(self, noteOrRest):
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
      return pd.NA
    return noteOrRest

  def m21ObjectsNoTies(self):
    if 'm21ObjectsNoTies' not in self.analyses:
      self.analyses['m21ObjectsNoTies'] = self.m21_objects().applymap(self._remove_tied).dropna(how='all')
    return self.analyses['m21ObjectsNoTies']

  def midi_pitches(self):
    '''\tProcess notes as midi pitches. Midi does not have a representation
    for rests, so use -1 as a placeholder.'''
    if 'midi_pitches' not in self.analyses:
      midi_pitches = self.m21ObjectsNoTies().applymap(lambda noteOrRest: -1 if noteOrRest.isRest else noteOrRest.pitch.midi, na_action='ignore')
      midi_pitches = midi_pitches.ffill().astype(int)
      self.analyses['midi_pitches'] = midi_pitches
    return self.analyses['midi_pitches']


  def piano_roll(self):
    '''\tConstruct midi piano roll. NB: there are 128 possible midi pitches.'''
    if 'piano_roll' not in self.analyses:
      piano_roll = pd.DataFrame(index=range(128), columns=self.midi_pitches().index.values)
      for offset in self.midi_pitches().index:
        for pitch in self.midi_pitches().loc[offset]:
          if pitch >= 0:
            piano_roll.at[pitch, offset] = 1
      piano_roll.fillna(0, inplace=True)
      self.analyses['piano_roll'] = piano_roll
    return self.analyses['piano_roll']

  def sampled(self, winms=100, sample_rate=2000, bpm=60):
    '''\tSample the score according to bpm, sample_rate, and winms.'''
    key = ('sampled', winms, sample_rate, bpm)
    if key not in self.analyses:
      # freqs = [round(2**((i-69)/12) * aFreq, 3) for i in range(128)] 
      # piano_roll.index = freqs
      pr_cols = self.piano_roll().columns
      col_set = set(pr_cols)  # this set makes sure that any timepoints in piano_roll cols will persist
      slices = 60/bpm * 20
      col_set.update([t/slices for t in range(0, int(self.score.highestTime * slices))])
      sampled = pd.DataFrame(columns=sorted(col_set), index=self.piano_roll().index)
      sampled.update(self.piano_roll())
      sampled = sampled.ffill(axis=1).fillna(0)
      self.analyses[key] = sampled
    return self.analyses[key]

  def mask(self, winms=100, sample_rate=2000, num_harmonics=1, width=0,
            bpm=60, aFreq=440, base_note=0, tuning_factor=1,
            append_function=True, append_harm=True):
    '''\tConstruct a mask from the sampled piano roll using width and harmonics.'''
    key = ('mask', winms, sample_rate, num_harmonics, width, bpm, aFreq, base_note, tuning_factor)
    if key not in self.analyses:
      width_semitone_factor = 2 ** ((width / 2) / 12)
      sampled = self.sampled(winms, sample_rate, bpm)
      num_rows = int(2 ** round(math.log(winms / 1000 * sample_rate) / math.log(2) - 1)) + 1
      mask = pd.DataFrame(index=range(num_rows), columns=sampled.columns).fillna(0)
      fftlen = 2**round(math.log(winms / 1000 * sample_rate) / math.log(2))

      for row in range(base_note, sampled.shape[0]):
        note = base_note + row
        # MIDI note to Hz: MIDI 69 = 440 Hz = A4
        freq = tuning_factor * (2 ** (note / 12)) * aFreq / (2 ** (69 / 12))
        if sampled.loc[row, :].sum() > 0:
          mcol = pd.Series(0, index=range(num_rows))
          for harm in range(1, num_harmonics + 1):
            minbin = math.floor(harm * freq / width_semitone_factor / sample_rate * fftlen)
            maxbin = math.ceil(harm * freq * width_semitone_factor / sample_rate * fftlen)
            if minbin <= num_rows:
              maxbin = min(maxbin, num_rows)
              mcol.loc[minbin : maxbin] = 1
          mask.iloc[np.where(mcol)[0], np.where(sampled.iloc[row])[0]] = 1
      self.analyses[key] = mask
    ret = self.analyses[key].copy()
    if append_function:
      func = self.functions()
      if func.empty:
        print('No **function spine found, returning mask without function analysis.')
      else:
        ret.loc['Function'] = func.reindex_like(ret.iloc[0]).ffill()
    if append_harm:
      harm = self.harmonies()
      if harm.empty:
        print('No **harm spine found, returning mask without harmonic analysis.')
      else:
        ret.loc['Harmony'] = harm.reindex_like(ret.iloc[0]).ffill()
    return ret


# piece = Score(score_path='./test_files/mozart.krn')
# harm = piece.harmonies()
# pdb.set_trace()
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
