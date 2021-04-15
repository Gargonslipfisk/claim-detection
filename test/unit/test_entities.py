import pytest
from json import load
from enum import Enum
import os
from configparser import ConfigParser

from src.utils.singleton import SingletonMetaclass
from src.utils.entities import decorated_extract_entity_es

# -----------------------------------------------------------------
parent_path = os.path.abspath(__file__ + "/../../")

@pytest.fixture
def cfg():
    # Load a config
    cfg = ConfigParser(allow_no_value=True)
    cfg.read(parent_path + '/cfg/test_config.cfg')
    yield cfg
    # Remove the config object and clear it from the singleton cache
    del cfg
    SingletonMetaclass.clear_instances()

class TestSets(Enum):
    entidades = 'entidades'

def from_json(path):
    with open(path, 'r', encoding='utf8') as rd:
        data = load(rd)
    return data

testdata = [
    ([(1, ['ETA', 'ORG']),
      (1, ['WebEx', 'PER']),
      (1, ['PP', 'ORG']),
      (0, [])])]

@pytest.mark.parametrize('out', testdata)
def test_decorated_extract_entity_es(cfg, out):
    clave = TestSets.entidades
    test_data_file = os.path.join(parent_path, cfg['TEST']['test_data_folder'], cfg['TEST']['test_data_filename'])
    utterances = from_json(test_data_file)
    actual = [decorated_extract_entity_es(u) for u in utterances[clave.value]]
    assert actual == out
