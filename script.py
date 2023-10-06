import pandas as pd
import numpy as np
import music21 as m21
import math
import ast
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
    self.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    self.fileExtension = score_path.rsplit('.', 1)[1]
    self.metadata = {'Title': self.score.metadata.title, 'Composer': self.score.metadata.composer}
    self._partStreams = self.score.getElementsByClass(m21.stream.Part)
    self._semiFlatParts = [part.semiFlat for part in self._partStreams]
    self.partNames = []
    self.public = '\n'.join([f'{prop.ljust(15)}{type(getattr(self, prop))}' for prop in dir(self) if not prop.startswith('_')])
    self._analyses = {}
    for i, part in enumerate(self._semiFlatParts):
      name = part.partName if (part.partName and part.partName not in self.partNames) else 'Part_' + str(i + 1)
      self.partNames.append(name)
    self.parts = []
    for i, flat_part in enumerate(self._semiFlatParts):
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
      if len(df.columns) > 1:
        df.columns = [':'.join((self.partNames[i], str(j))) for j in range(1, len(df.columns) + 1)]
      else:
        df.columns = [self.partNames[i]]
      # for now remove multiple events at the same offset in a given part
      df = df[~df.index.duplicated(keep='last')]
      self.parts.append(df)
    self._import_function_harm_spines()
  
  def _import_function_harm_spines(self):
    if self.fileExtension == 'krn':
      humFile = m21.humdrum.spineParser.HumdrumFile(self.path)
      humFile.parseFilename()
      objs = self._m21_objects()
      for spine in humFile.spineCollection:
        if spine.spineType in ('harm', 'function', 'cdata'):
          start = False
          vals, valPositions = [], []
          if spine.spineType == 'harm':
            keyVals, keyPositions = [], []
          for i, event in enumerate(spine.eventList):
            contents = event.contents
            if contents.endswith(':') and contents.startswith('*'):
              start = True
              # there usually won't be any m21 objects at the same position as the key events,
              # so use the position from the next item in eventList if there is a next item.
              if spine.spineType == 'harm' and i + 1 < len(spine.eventList):
                keyVals.append(contents)
                keyPositions.append(spine.eventList[i+1].position)
              continue
            elif not start or '!' in contents or '=' in  contents or '*-' == contents:
              continue
            else:
              vals.append(contents)
              valPositions.append(event.position)

          df1 = self._priority()
          name = spine.spineType.title()
          if name == 'Cdata':
            df2 = pd.DataFrame([ast.literal_eval(val) for val in vals], index=valPositions)
          else:
            df2 = pd.DataFrame({name: vals}, index=valPositions)
          joined = df1.join(df2, on='Priority')
          res = joined.iloc[:, 2:].copy()  # get all the columns from the third to the end. Usually just 1 col except for cdata
          res.index = joined['Offset']
          res.index.name = ''
          self._analyses[(spine.spineType, 0)] = res
          if spine.spineType == 'harm' and len(keyVals):
            keyName = 'harmKeys'
            df3 = pd.DataFrame({keyName: keyVals}, index=keyPositions)
            joined = df1.join(df3, on='Priority')
            df3 = joined.iloc[:, 2:].copy()
            df3.index = joined['Offset']
            df3.index.name = ''
            self._analyses[(keyName, 0)] = df3

    if ('function', 0) not in self._analyses:
      self._analyses[('function', 0)] = pd.DataFrame()
    if ('harm', 0) not in self._analyses:
      self._analyses[('harm', 0)] = pd.DataFrame()
    if ('harmKeys', 0) not in self._analyses:
      self._analyses[('harmKeys', 0)] = pd.DataFrame()
  
  def _m21_objects(self):
    if '_m21_objects' not in self._analyses:
      self._analyses['_m21_objects'] = pd.concat(self.parts, axis=1, sort=True)
    return self._analyses['_m21_objects']
  
  def lyrics(self):
    if 'lyrics' not in self._analyses:
      self._analyses['lyrics'] = self._m21_objects().applymap(lambda cell: cell.lyric or np.nan, na_action='ignore').dropna(how='all')
    return self._analyses['lyrics']

  def _priority(self):
    '''\tFor .krn files, get the line numbers of the events in the piece, which music21
    often calls "priority". For other encoding formats return an empty dataframe.'''
    if '_priority' in self._analyses:
      return self._analyses['_priority']
    if self.fileExtension != 'krn':
      priority = pd.DataFrame()
    else:
      priority = self._m21_objects().applymap(lambda cell: cell.priority, na_action='ignore').ffill(axis=1).iloc[:, -1].astype(int)
      priority = pd.DataFrame({'Priority': priority.values, 'Offset': priority.index})
    self._analyses['_priority'] = priority
    return priority

  def _reindex_like_sampled(self, df, bpm=60, obs=20):
    '''\tGiven a pandas.DataFrame, reindex it like the columns of the piano roll sampled
    at `obs` observations a second at the given bpm assuming the quarter note is the beat.
    If an index value in the passed dataframe is not in the columns of the sampled piano
    roll, it will be rewritten to the nearest preceding index val and if this creates
    duplicate index values, only the last one will be kept. Returns a forward-filled 
    copy of the passed dataframe. The original dataframe is not changed.'''
    _df = df.copy()
    timepoints = self.sampled(bpm, obs).iloc[0, :]
    ndx = [val if val in timepoints.index else timepoints.index.asof(val) for val in _df.index]
    _df.index = ndx
    _df = _df[~_df.index.duplicated(keep='last')]
    _df = _df.reindex(index=timepoints.index).ffill()
    return _df

  def harmKeys(self, bpm=0):
    '''\tGet the keys the **harm spine is done in as a pandas series if this piece
    is a kern file and has a **harm spine. Otherwise return an empty series. The
    index of the series will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0, it will return the key observations from
    the **harm spine at their original offsets.'''
    key = ('harmKeys', bpm)
    if key not in self._analyses:
      harmKeys = self._analyses[('harmKeys', 0)]
      self._analyses[key] = self._reindex_like_sampled(harmKeys, bpm)
    return self._analyses[key]

  def harmonies(self, bpm=0):
    '''\tGet the harmonic analysis from the **harm spine as a pandas series if this
    piece is a kern file and has a **harm spine. Otherwise return an empty series. The
    index of the series will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0, it will return the **harm spine
    observations at their original offsets.'''
    key = ('harm', bpm)
    if key not in self._analyses:
      harm = self._analyses[('harm', 0)]
      self._analyses[key] = self._reindex_like_sampled(harm, bpm)
    return self._analyses[key]

  def functions(self, bpm=0):
    '''\tGet the functional analysis from the **function spine as a pandas series if this
    piece is a kern file and has a **function spine. Otherwise return an empty series. The
    index of the series will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0, it will return the **function spine
    observations at their original offsets.'''
    key = ('function', bpm)
    if key not in self._analyses:
      functions = self._analyses[('function', 0)]
      self._analyses[key] = self._reindex_like_sampled(functions, bpm)
    return self._analyses[key]

  def _remove_tied(self, noteOrRest):
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
      return np.nan
    return noteOrRest

  def _m21ObjectsNoTies(self):
    if '_m21ObjectsNoTies' not in self._analyses:
      self._analyses['_m21ObjectsNoTies'] = self._m21_objects().applymap(self._remove_tied).dropna(how='all')
    return self._analyses['_m21ObjectsNoTies']

  def durations(self):
    '''\tReturn dataframe of durations of note and rest objects in piece.'''
    if 'durations' not in self._analyses:
      m21objs = self._m21ObjectsNoTies()
      sers = []
      for col in range(len(m21objs.columns)):
        part = m21objs.iloc[:, col].dropna()
        if len(part) > 1:
          vals = (part.index[1:] - part.index[:-1]).to_list()
        else:
          vals = []
        vals.append(self.score.highestTime - part.index[-1])
        sers.append(pd.Series(vals, part.index))
      df = pd.concat(sers, axis=1, sort=True)
      self._analyses['durations'] = df
      df.columns = m21objs.columns
    return self._analyses['durations']

  def midiPitches(self):
    '''\tReturn a dataframe of notes and rests as midi pitches. Midi does not
    have a representation for rests, so -1 is used as a placeholder.'''
    if 'midiPitches' not in self._analyses:
      midiPitches = self._m21ObjectsNoTies().applymap(lambda noteOrRest: -1 if noteOrRest.isRest else noteOrRest.pitch.midi, na_action='ignore')
      self._analyses['midiPitches'] = midiPitches
    return self._analyses['midiPitches']

  def nmats(self, bpm=60):
    '''\tReturn a dictionary of dataframes, one for each voice, each with the following
    columns about the notes and rests in that voice:

    ONSET_BEAT    DURATION_BEAT    PART    MIDI    ONSET_SEC    OFFSET_SEC

    In the MIDI column, notes are represented with their midi pitch numbers 0 to 127
    inclusive, and rests are represented with -1s. The ONSET and OFFSET columns given
    in seconds are directly proportional to the ONSET_BEATS column and ONSET_BEATS +
    DURATION_BEATS columns respectively. The proportion used is determined by the `bpm`
    argument.'''
    key = ('nmats', bpm)
    if key not in self._analyses:
      nmats = {}
      dur = self.durations()
      mp = self.midiPitches()
      toSeconds = 60/bpm
      for i, partName in enumerate(self.partNames):
        midi = mp.iloc[:, i].dropna()
        onsetBeat = midi.index.to_series()
        durBeat = dur.iloc[:, i].dropna()
        part = pd.Series(partName, midi.index)
        onsetSec = onsetBeat * toSeconds
        offsetSec = (onsetBeat + durBeat) * toSeconds
        df = pd.concat([onsetBeat, durBeat, part, midi, onsetSec, offsetSec], axis=1)
        df.columns = ['ONSET_BEAT', 'DURATION_BEAT', 'PART', 'MIDI', 'ONSET_SEC', 'OFFSET_SEC']
        nmats[partName] = df
      self._analyses[key] = nmats
    return self._analyses[key]

  def pianoRoll(self):
    '''\tConstruct midi piano roll. NB: there are 128 possible midi pitches.'''
    if 'pianoRoll' not in self._analyses:
      mp = self.midiPitches().ffill().astype(int)
      pianoRoll = pd.DataFrame(index=range(128), columns=mp.index.values)
      for offset in mp.index:
        for pitch in mp.loc[offset]:
          if pitch >= 0:
            pianoRoll.at[pitch, offset] = 1
      pianoRoll.fillna(0, inplace=True)
      self._analyses['pianoRoll'] = pianoRoll
    return self._analyses['pianoRoll']

  def sampled(self, bpm=60, obs=20):
    '''\tSample the score according to bpm, and the desired observations per second, `obs`.'''
    key = ('sampled', bpm, obs)
    if key not in self._analyses:
      slices = 60/bpm * obs
      timepoints = pd.Index([t/slices for t in range(0, int(self.score.highestTime * slices))])
      pr = self.pianoRoll().copy()
      pr.columns = [col if col in timepoints else timepoints.asof(col) for col in pr.columns]
      sampled = pr.reindex(columns=timepoints, method='ffill')
      self._analyses[key] = sampled
    return self._analyses[key]

  def mask(self, winms=100, sample_rate=2000, num_harmonics=1, width=0,
            bpm=60, aFreq=440, base_note=0, tuning_factor=1, obs=20):
    '''\tConstruct a mask from the sampled piano roll using width and harmonics.'''
    key = ('mask', winms, sample_rate, num_harmonics, width, bpm, aFreq, base_note, tuning_factor)
    if key not in self._analyses:
      width_semitone_factor = 2 ** ((width / 2) / 12)
      sampled = self.sampled(bpm, obs)
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
      self._analyses[key] = mask
    return self._analyses[key]
