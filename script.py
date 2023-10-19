import pandas as pd
import numpy as np
import music21 as m21
import math
import ast
import pdb
import json
import requests
import os
import tempfile
m21.environment.set('autoDownload', 'allow')

imported_scores = {}
_duration2Kern = {  # keys get rounded to 5 decimal places
  48: '000.',
  32: '000',
  24: '00.',
  16: '00',
  12: '0.',
  8: '0',
  6: '1.',
  4: '1',
  3: '2.',
  2.66666: '3%2',
  2: '2',
  1.5: '4.',
  1.33333: '3',
  1: '4',
  .75: '8.',
  .5: '8',
  .375: '16.',
  .25: '16',
  .1875: '32.',
  .125: '32',
  .09375: '64.',
  .0625: '64',
  .04688: '128.',
  .03125: '128',
  .02344: '256.',
  .01563: '256'
}

class Score:
  '''\tImport score via music21 and expose AMPACT's analysis utilities which are
  generally formatted as Pandas DataFrames.'''
  def __init__(self, score_path):
    self._analyses = {}
    self.path = score_path
    self._tempFile = ''
    self.fileName = score_path.rsplit('.', 1)[0].rsplit('/')[-1]
    self.fileExtension = score_path.rsplit('.', 1)[1]
    if score_path.startswith('http') and self.fileExtension == 'krn':
      fd, tmp_path = tempfile.mkstemp()
      try:
        with os.fdopen(fd, 'w') as tmp:
          response = requests.get(self.path)
          tmp.write(response.text)
          tmp.seek(0)
          self._assignM21Attributes(tmp_path)
          self._import_function_harm_spines(tmp_path)
      finally:
        os.remove(tmp_path)
    else:  # file is not an online kern file (can be either or neither but not both)
      self._assignM21Attributes()
      self._import_function_harm_spines()
    self.public = '\n'.join([f'{prop.ljust(15)}{type(getattr(self, prop))}' for prop in dir(self) if not prop.startswith('_')])
  
  def _assignM21Attributes(self, path=''):
    '''\tReturn a music21 score. This method is used internally for memoization purposes.'''
    if self.path not in imported_scores:
      if path:
        imported_scores[self.path] = m21.converter.parse(path, format='humdrum')
      else:
        imported_scores[self.path] = m21.converter.parse(self.path)
    self.score = imported_scores[self.path]
    self.metadata = {'Title': self.score.metadata.title, 'Composer': self.score.metadata.composer}
    self._partStreams = self.score.getElementsByClass(m21.stream.Part)
    self._semiFlatParts = [part.semiFlat for part in self._partStreams]

    self.partNames = []
    for i, part in enumerate(self._semiFlatParts):
      name = part.partName if (part.partName and part.partName not in self.partNames) else 'Part_' + str(i + 1)
      self.partNames.append(name)
    
    self.partDivisi = {}
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
        divisi = [':'.join((self.partNames[i], str(j))) for j in range(1, len(df.columns) + 1)]
      else:
        divisi = [self.partNames[i]]
      df.columns = divisi
      self.partDivisi[self.partNames[i]] = len(divisi)
      # for now remove multiple events at the same offset in a given part
      df = df[~df.index.duplicated(keep='last')]
      self.parts.append(df)

  def _parts(self):
    '''\tReturn a df of the note, rest, and chord objects in the score. The difference between
    parts and divisi is that parts can have chords whereas divisi cannot. If there are chords
    in the _parts df, the divisi df will include all these notes by adding additional columns.'''
    if '_parts' not in self._analyses:
      parts = []
      for i, flat_part in enumerate(self._semiFlatParts):
        ser = pd.Series(flat_part.getElementsByClass(['Note', 'Rest', 'Chord']), name=self.partNames[i])
        ser.index = ser.apply(lambda nrc: nrc.offset).round(4)
        ser = ser[~ser.index.duplicated(keep='last')]
        parts.append(ser)
      self._analyses['_parts'] = pd.concat(parts, axis=1, sort=True)
    return self._analyses['_parts']

  def _divisi(self):
    '''\tReturn a df of the note and rest objects in the score without chords. The difference between
    parts and divisi is that parts can have chords whereas divisi cannot. If there are chords
    in the _parts df, the divisi df will include all these notes by adding additional columns.'''
    if '_divisi' not in self._analyses:
      parts = self._parts()
      divisi = []
      for i, col in enumerate(parts.columns):
        part = parts[col].dropna()
        div = part.apply(lambda nrc: nrc.notes if nrc.isChord else (nrc,)).apply(pd.Series)
        if len(div.columns) > 1:
          div.columns = [':'.join((self.partNames[i], str(j))) for j in range(1, len(div.columns) + 1)]
        else:
          div.columns = [self.partNames[i]]
        divisi.append(div)
      self._analyses['_divisi'] = pd.concat(divisi, axis=1, sort=True)
    return self._analyses['_divisi']

  def _import_function_harm_spines(self, path=''):
    if self.fileExtension == 'krn' or path:
      humFile = m21.humdrum.spineParser.HumdrumFile(path or self.path)
      humFile.parseFilename()
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
            # pdb.set_trace()
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
    if ('cdata', 0) not in self._analyses:
      self._analyses[('cdata', 0)] = pd.DataFrame()

  def lyrics(self):
    if 'lyrics' not in self._analyses:
      self._analyses['lyrics'] = self._divisi().applymap(lambda cell: cell.lyric or np.nan, na_action='ignore').dropna(how='all')
    return self._analyses['lyrics']

  def _priority(self):
    '''\tFor .krn files, get the line numbers of the events in the piece, which music21
    often calls "priority". For other encoding formats return an empty dataframe.'''
    if '_priority' not in self._analyses:
      if self.fileExtension != 'krn':
        priority = pd.DataFrame()
      else:
        priority = self._parts().applymap(lambda cell: cell.priority, na_action='ignore').ffill(axis=1).iloc[:, -1].astype('Int16')
        priority = pd.DataFrame({'Priority': priority.values, 'Offset': priority.index})
      self._analyses['_priority'] = priority
    return self._analyses['_priority']

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
    '''\tGet the keys the **harm spine is done in as a pandas dataframe if this piece
    is a kern file and has a **harm spine. Otherwise return an empty dataframe. The
    index of the dataframe will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0 (default), it will return the key observations from
    the **harm spine at their original offsets.'''
    key = ('harmKeys', bpm)
    if key not in self._analyses:
      harmKeys = self._analyses[('harmKeys', 0)]
      self._analyses[key] = self._reindex_like_sampled(harmKeys, bpm)
    return self._analyses[key]

  def harmonies(self, bpm=0):
    '''\tGet the harmonic analysis from the **harm spine as a pandas dataframe if this
    piece is a kern file and has a **harm spine. Otherwise return an empty dataframe. The
    index of the dataframe will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0 (default), it will return the **harm spine
    observations at their original offsets.'''
    key = ('harm', bpm)
    if key not in self._analyses:
      harm = self._analyses[('harm', 0)]
      self._analyses[key] = self._reindex_like_sampled(harm, bpm)
    return self._analyses[key]

  def functions(self, bpm=0):
    '''\tGet the functional analysis from the **function spine as a pandas dataframe if this
    piece is a kern file and has a **function spine. Otherwise return an empty dataframe. The
    index of the dataframe will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0 (default), it will return the **function spine
    observations at their original offsets.'''
    key = ('function', bpm)
    if key not in self._analyses:
      functions = self._analyses[('function', 0)]
      self._analyses[key] = self._reindex_like_sampled(functions, bpm)
    return self._analyses[key]

  def cdata(self, bpm=0):
    '''\tGet the cdata analysis from the **cdata spine as a pandas dataframe if this
    piece is a kern file and has a **cdata spine. Otherwise return an empty dataframe. The
    index of the dataframe will match the columns of the sampled piano roll created with the
    same bpm as that passed. If bpm is set to 0 (default), it will return the **cdata spine
    observations at their original offsets.'''
    key = ('cdata', bpm)
    if key not in self._analyses:
      cdata = self._analyses[('cdata', 0)]
      self._analyses[key] = self._reindex_like_sampled(cdata, bpm)
    return self._analyses[key]

  def _remove_tied(self, noteOrRest):
    if hasattr(noteOrRest, 'tie') and noteOrRest.tie is not None and noteOrRest.tie.type != 'start':
      return np.nan
    return noteOrRest

  def _m21ObjectsNoTies(self):
    if '_m21ObjectsNoTies' not in self._analyses:
      self._analyses['_m21ObjectsNoTies'] = self._divisi().applymap(self._remove_tied).dropna(how='all')
    return self._analyses['_m21ObjectsNoTies']

  def _measures(self, divisi=True):
    '''\tReturn df of the measure starting points.'''
    key = ('_measure', divisi)
    if key not in self._analyses:
      partMeasures = []
      for i, part in enumerate(self._semiFlatParts):
        ser = pd.Series({m.offset: m.measureNumber for m in part.getElementsByClass(['Measure'])}, dtype='Int16')
        if divisi:
          partMeasures.extend([ser] * self.partDivisi[self.partNames[i]])
        else:
          partMeasures.append(ser)
      df = pd.concat(partMeasures, axis=1)
      df.columns = self._divisi().columns if divisi else self.partNames
      self._analyses[key] = df
    return self._analyses[key]

  def _barlines(self):
    '''\tReturn df of barlines specifying which barline type. Double barline, for
    example, can help detect section divisions, and the final barline can help
    process the `highestTime` similar to music21.'''
    if "_barlines" not in self._analyses:
      partBarlines = []
      for i, part in enumerate(self._semiFlatParts):
        ser = pd.Series({b.offset: b.type for b in part.getElementsByClass(['Barline'])})
        partBarlines.extend([ser] * self.partDivisi[self.partNames[i]])
      df = pd.concat(partBarlines, axis=1)
      df.columns = self._divisi().columns
      self._analyses["_barlines"] = df
    return self._analyses["_barlines"]

  def _keySignatures(self):
    if '_keySignatures' not in self._analyses:
      kSigs = []
      for i, part in enumerate(self._semiFlatParts):
        kSigs.append(pd.Series({ms.offset: ''.join([_note.name for _note in ms.keySignature.alteredPitches]).lower()
                                for ms in part.getElementsByClass(['Measure']) if ms.keySignature is not None}, name=self.partNames[i]))
      df = pd.concat(kSigs, axis=1).sort_index(kind='mergesort')
      self._analyses['_keySignatures'] = df
    return self._analyses['_keySignatures']

  def _timeSignatures(self):
    if '_timeSignatures' not in self._analyses:
      tsigs = []
      for i, part in enumerate(self._semiFlatParts):
        tsigs.append(pd.Series({ts.offset: ts.ratioString for ts in part.getTimeSignatures()}, name=self.partNames[i]))
      df = pd.concat(tsigs, axis=1).sort_index(kind='mergesort')
      self._analyses['_timeSignatures'] = df
    return self._analyses['_timeSignatures']

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

  def _noteRestHelper(self, noteOrRest):
    if noteOrRest.isRest:
      return 'r'
    return noteOrRest.nameWithOctave

  def _combineRests(self, col):
      col = col.dropna()
      return col[(col != 'r') | ((col == 'r') & (col.shift(1) != 'r'))]

  def _combineUnisons(self, col):
      col = col.dropna()
      return col[(col == 'r') | (col != col.shift(1))]

  def notes(self, combine_rests=True, combine_unisons=False):
    '''\tReturn a dataframe of the notes and rests given in American Standard Pitch
    Notation where middle C is C4. Rests are designated with the string "r".

    If `combine_rests` is True (default), non-first consecutive rests will be
    removed, effectively combining consecutive rests in each voice.
    `combine_unisons` works the same way for consecutive attacks on the same
    pitch in a given voice, however, `combine_unisons` defaults to False.'''
    if 'notes' not in self._analyses:
      df = self._m21ObjectsNoTies().applymap(self._noteRestHelper, na_action='ignore')
      self._analyses['notes'] = df
    ret = self._analyses['notes'].copy()
    if combine_rests:
      ret = ret.apply(self._combineRests)
    if combine_unisons:
      ret = ret.apply(self._combineUnisons)
    return ret

  def _kernNoteHelper(self, _note):
    '''\tParse a music21 note object into a kern note token.'''
    # TODO: this doesn't seem to be detecting longas in scores. Does m21 just not detect longas in kern files? Test with mei, midi, and xml
    dur = _duration2Kern[round(_note.quarterLength, 5)]
    _oct = _note.octave
    if _oct > 3:
      step = _note.step.lower() * (_oct - 3)
    else:
      step = _note.step * (4 - _oct)
    acc = _note.pitch.accidental
    acc = acc.modifier if acc is not None else ''
    longa = 'l' if _note.duration.type == 'longa' else ''
    return f'{dur}{step}{acc}{longa}'

  def _kernChordHelper(self, _chord):
    '''\tParse a music21 chord object into a kern chord token.'''
    # TODO: figure out how durations are handled in kern chords. Might need to pass the chord's duration down to this func since m21 pitch objects don't have duration attributes
    pitches = []
    dur = _duration2Kern[round(_chord.quarterLength, 5)]
    for _pitch in _chord.pitches:
      _oct = _pitch.octave
      if _oct > 3:
        letter = _pitch.step.lower() * (_oct - 3)
      else:
        letter = _pitch.step * (4 - _oct)
      acc = _pitch.accidental
      acc = acc.modifier if acc is not None else ''
      longa = '' #'l' if _pitch.duration.type == 'longa' else ''
      pitches.extend((dur, letter, acc, longa, ' '))
    if len(pitches):
      return ''.join(pitches[:-1])
    else:
      return ''

  def _kernNRCHelper(self, nrc):
    '''\tConvert a music21 note, rest, or chord object to its corresponding kern token.'''
    if nrc.isNote:
      return self._kernNoteHelper(nrc)
    elif nrc.isRest:
      return f'{_duration2Kern[round(nrc.quarterLength, 5)]}r'
    else:
      return self._kernChordHelper(nrc)

  def kernNotes(self):
    '''\tReturn a dataframe of the notes and rests given in kern notation. This is
    not the same as creating a kern format of a score, but is an important step
    in that process.'''
    if 'kernNotes' not in self._analyses:
      parts = self._parts()
      df = parts.applymap(self._kernNRCHelper, na_action='ignore')
      self._analyses['kernNotes'] = df
    return self._analyses['kernNotes'].copy()

  def nmats(self, bpm=60):
    '''\tReturn a dictionary of dataframes, one for each voice, each with the following
    columns about the notes and rests in that voice:

    MEASURE    ONSET_BEAT    DURATION_BEAT    PART    MIDI    ONSET_SEC    OFFSET_SEC

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
      ms = self._measures()
      toSeconds = 60/bpm
      for i, partName in enumerate(self.partNames):
        meas = ms.iloc[:, i]
        midi = mp.iloc[:, i].dropna()
        onsetBeat = midi.index.to_series()
        durBeat = dur.iloc[:, i].dropna()
        part = pd.Series(partName, midi.index)
        onsetSec = onsetBeat * toSeconds
        offsetSec = (onsetBeat + durBeat) * toSeconds
        df = pd.concat([meas, onsetBeat, durBeat, part, midi, onsetSec, offsetSec], axis=1)
        df.columns = ['MEASURE', 'ONSET_BEAT', 'DURATION_BEAT', 'PART', 'MIDI', 'ONSET_SEC', 'OFFSET_SEC']
        nmats[partName] = df
      self._analyses[key] = nmats
    return self._analyses[key]

  def pianoRoll(self):
    '''\tConstruct midi piano roll. NB: there are 128 possible midi pitches.'''
    if 'pianoRoll' not in self._analyses:
      mp = self.midiPitches().ffill()
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

  def fromJSON(self, json_path):
    '''\tWIP: Return a pandas dataframe of the JSON file. The outermost keys will get
    interpretted as the index values of the table, and the second-level keys
    will be the columns.'''
    with open(json_path) as json_data:
      data = json.load(json_data)
    df = pd.DataFrame(data).T
    df.index = pd.DatetimeIndex(df.index)
    return df
  
  def _kernHeader(self):
    '''\tReturn a string of the kern format header global comments.'''
    data = [
      f'!!!COM: {self.metadata["Composer"] or "Composer not found"}',
      f'!!!OTL: {self.metadata["Title"] or "Title not found"}'
    ]
    return '\n'.join(data)
    # f'!!!voices: {len(cols)}', 
    # ['**kern'] * len(cols),

  def _kernFooter(self):
    '''Return a string of the kern format footer global comments.'''
    from datetime import datetime
    data = [
      '!!!RDF**kern: %=rational rhythm',
      '!!!RDF**kern: l=long note in original notation',
      '!!!RDF**kern: i=editorial accidental',
      f'!!!ONB: Translated from {self.fileExtension} file on {datetime.today().strftime("%Y-%m-%d")} via AMPACT'
    ]
    if 'Title' in self.metadata:
      data.append('!!!title: @{OTL}')
    return '\n'.join(data)

  def toKern(self, path_name='', data=''):
    '''\t*** WIP: currently not outputting valid kern files. ***
    Create a kern representation of the score. If no `path_name` variable is
    passed, then returns a pandas DataFrame of the kern representation. Otherwise
    a file is created or overwritten at the `path_name` path. If path_name does not
    end in '.krn' then this file extension will be added to the path.'''
    key = ('toKern', data)
    if key not in self._analyses:
      _me = self._measures()
      me = _me.astype('string').applymap(lambda cell: '=' + cell + '-' if cell == '0' else '=' + cell, na_action='ignore')
      events = self.kernNotes()
      events = events[reversed(events.columns)]
      ba = self._barlines()
      ba = ba[ba != 'regular'].dropna().replace({'double': '||', 'final': '=='})
      ba.loc[self.score.highestTime, :] = '=='
      if data:
        cdata = self.fromJSON(data)
        cdata.index = cdata.index.second
        firstTokens = ['**kern'] * len(events.columns) + ['**data'] * len(cdata.columns)
        instruments = ['*Ivox'] * len(events.columns) + ['*'] * len(cdata.columns)
        partNames = [f'*I"{name}' for name in events.columns] + [f'*{col}' for col in cdata.columns]
        shortNames = [f"*I'{name[0]}" for name in events.columns] + ['*'] * len(cdata.columns)
        events = pd.concat([events, cdata], axis=1)
      else:
        firstTokens = ['**kern'] * len(events.columns)
        instruments = ['*Ivox'] * len(events.columns)
        partNames = [f'*I"{name}' for name in events.columns]
        shortNames = [f"*I'{name[0]}" for name in events.columns]
      me = pd.concat([me.iloc[:, 0]] * len(events.columns), axis=1)
      ba = pd.concat([ba.iloc[:, 0]] * len(events.columns), axis=1)
      me.columns = events.columns
      ba.columns = events.columns
      ts = ('*M' + self._timeSignatures())
      ts = ts.reindex(events.columns, axis=1).fillna('*')
      ks = '*k[' + self._keySignatures() + ']'
      ks = ks.reindex(events.columns, axis=1).fillna('*')
      partTokens = pd.DataFrame([firstTokens, instruments, partNames, shortNames, ['*-']*len(events.columns)],
                                index=[-10, -9, -8, -7, int(self.score.highestTime + 1)])
      partTokens.columns = events.columns
      body = pd.concat([partTokens, me, ks, ts, events, ba]).sort_index(kind='mergesort').fillna('.')
      body = body.to_csv(sep='\t', header=False, index=False, quotechar='`')
      result = ''.join([self._kernHeader(), '\n', body, self._kernFooter()])
      self._analyses[key] = result
    if not path_name:
      return self._analyses[key]
    else:
      if not path_name.endswith('.krn'):
        path_name += '.krn'
      if '/' not in path_name:
        path_name = './output_files/' + path_name
      with open(path_name, 'w') as f:
        f.write(self._analyses[key])
