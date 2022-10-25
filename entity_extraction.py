from collections import Counter, defaultdict
import random

import utilities, wikidata

ignore_ner_labels = ['MONEY', 'NUMBER', 'ORDINAL', 'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET', 'O']

PERSON_TYPE = 'PERSON'
LOCATION_TYPE = 'LOCATION'
ORGANIZATION_TYPE = 'ORGANIZATION'
MISC_TYPE = 'MISC'
NAMED_TYPES = set([PERSON_TYPE, LOCATION_TYPE, ORGANIZATION_TYPE, MISC_TYPE])

class EntityExtractor:
 
    def __init__(self):
        self._entity_to_id = {}
        self._occurances = {}
        self._occur_count = Counter()
        self._next_id = random.randint(0, 10000)

    @property
    def ids(self): return self._entity_to_id.items()

    @property
    def occurances(self): return self._occurances

    def most_common(self, N=1): return self._occur_count.most_common(N)

    def create_entity(self, entity, entity_id=None, dangermode=False):
        if dangermode is not True: assert entity not in self._entity_to_id, '"{}" already present in entity to id mapping'.format(str(entity))
            
        _, text = entity

        if entity_id is None:
            wikidata_id = wikidata.get_id(text)
            if wikidata_id: entity_id = ('wikidata', wikidata_id)
            else:
                entity_id = ('manual', self._next_id)
                self._next_id = random.randint(0, 10000)
                
        self._entity_to_id[entity] = entity_id

        if entity_id not in self._occurances: self._occurances[entity_id] = set()

    def add_occurance(self, entity, sent_idx, start_idx, end_idx): self.add_occurances(entity, {(sent_idx, start_idx, end_idx)})

    def add_occurances(self, entity, occurs_iter):
        if entity not in self._entity_to_id: self.create_entity(entity)
        eid = self._entity_to_id[entity]
        occurs = set(occurs_iter)
        self._occurances[eid].update(occurs)
        self._occur_count[eid] += len(occurs)

    def merge_entities(self, eid_dest, eid_additional):
        self._occurances[eid_dest] |= self._occurances[eid_additional]
        self._occur_count = len(self._occurances[eid_dest])

        keys_to_change = set(entity_key for entity_key, eid in self.ids if eid == eid_additional)
        for key in keys_to_change:
            self._entity_to_id[key] = eid_dest
            if eid_additional in self.occurances: del self.occurances[eid_additional]
                
    @staticmethod
    def from_sentences_with_entity_ids_2(sentences):
        ee = EntityExtractor()
        for sentence in sentences:
            
            start_idx = curr_eid = None
            for token in sentence['tokens']:
                
                if tuple(token.get('entity_id', ())) != curr_eid and curr_eid is not None:
                    end_idx = token['index'] - 1
                    raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                    if curr_eid not in ee._occurances or raw_text not in ee._entity_to_id: ee.create_entity(('BLANK', raw_text), curr_eid, dangermode=True)
                    ee.add_occurance(('BLANK', raw_text), sentence['index'], start_idx, end_idx)
                    start_idx = curr_eid = None
                    
                if 'entity_id' in token and curr_eid is None:
                    start_idx = token['index'] - 1
                    curr_eid = tuple(token['entity_id'])
                    
            if curr_eid is not None:
                end_idx = token['index']
                raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                ee.add_occurance(('BLANK', raw_text), sentence['index'], start_idx, end_idx)
                
        return ee
    
    @staticmethod
    def from_sentences_with_entity_ids(sentences):
        ee = EntityExtractor()
        for sentence in sentences:
            
            start_idx = curr_eid = None
            for token in sentence['tokens']:
                
                if tuple(token.get('entity_id', ())) != curr_eid and curr_eid is not None:
                    end_idx = token['index'] - 1
                    raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                    if curr_eid not in ee._occurances or raw_text not in ee._entity_to_id: ee.create_entity(('BLANK', raw_text), curr_eid, dangermode=True)
                    ee.add_occurance(('BLANK', raw_text), sentence['index'], start_idx, end_idx)
                    start_idx = curr_eid = None
                    
                if 'entity_id' in token and curr_eid is None:
                    start_idx = token['index'] - 1
                    curr_eid = tuple(token['entity_id'])
                    
            if curr_eid is not None:
                end_idx = token['index']
                raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                ee.add_occurance(('BLANK', raw_text), sentence['index'], start_idx, end_idx)
                
        return ee

    @staticmethod
    def from_sentences(sentences):
        ee = EntityExtractor()
        for sentence in sentences:
            
            start_idx = curr_type = None
            for token in sentence['tokens']:
                
                if token['ner'] != curr_type and curr_type is not None:
                    end_idx = token['index'] - 1
                    raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                    ee.add_occurance((curr_type, raw_text), sentence['index'], start_idx, end_idx)
                    start_idx = curr_type = None
                    
                if token['ner'] not in ignore_ner_labels and curr_type is None:
                    start_idx = token['index'] - 1
                    curr_type = token['ner']
                    
            if curr_type is not None:
                end_idx = token['index']
                raw_text = utilities.get_text(sentence['tokens'][start_idx:end_idx])
                ee.add_occurance((curr_type, raw_text), sentence['index'], start_idx, end_idx)
                
        return ee

    @staticmethod
    def from_parsed_data(entity_occurances):
        ee = EntityExtractor()
        for entity, occurances in entity_occurances.items():
            for sent_idx, e_type, start_idx, end_idx in occurances:
                if e_type not in ignore_ner_labels:
                    ee.add_occurance((e_type, entity), sent_idx - 1, start_idx - 1, end_idx)
                    
        return ee

def merge_people_by_last_name(ee):
    people = list(filter(lambda pair: pair[0][0] == PERSON_TYPE, ee.ids))
    people.sort(key=lambda pair: (pair[0][1].split(' ')[-1], len(pair[0][1])))
    potential_merges = set()
    curr_last_name = None
    curr_last_name_eid = None

    for (_, name), eid in people:
        
        if len(name.split(' ')) == 1:
            curr_last_name = name
            curr_last_name_eid = eid
        elif curr_last_name is not None:
            if curr_last_name == name.split(' ')[-1]:
                potential_merges.add((curr_last_name_eid, eid))

    count = Counter(eid for pair in potential_merges for eid in pair)
    merges = [pair for pair in potential_merges if count[pair[0]] == 1 and count[pair[1]] == 1]

    for eid_dest, eid_additional in merges: ee.merge_entities(eid_dest, eid_additional)

def occurs_by_sentence(ee):
    mapping = defaultdict(set)
    for eid, occurs in ee.occurances.items():
        for occur in occurs: mapping[occur[0]].add((occur, eid))
            
    return mapping