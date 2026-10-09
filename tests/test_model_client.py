from unittest.mock import Mock, patch
import pytest
import requests

from core import SCENARIOS
from model_client import (
    DecisionModelError, create_payload, local_server_url, parse_decision, probe_server, run_inference,
)

P = dict.fromkeys(['bearing_wear','lubrication','misalignment','imbalance','normal','uncertain'], 0.1)
P['bearing_wear'] = 0.5


def sample_reply():
    return {'model':'kev-4b', 'answers': {
        'fault': {'type':'choice','choice':'bearing_wear','probabilities':dict(P),'confidence':0.42},
        'severity': {'type':'score','score':2.2,
                     'probabilities':{'0':0.05,'1':0.10,'2':0.55,'3':0.20,'4':0.10}},
        'inspect': {'type':'noul','noul':0.92},
    }, 'usage':{'input_tokens':235}}


def test_systemone_request_contract():
    payload = create_payload(SCENARIOS['bearing'].state)
    assert set(payload['questions']) == {'fault','severity','inspect'}
    assert [payload['questions'][q]['type'] for q in ('fault','severity','inspect')] == ['choice','score','noul']
    assert isinstance(payload['state'], dict)
    assert payload['state']['sensors']['bearing_temperature_c'] == 84
    assert 'model' not in payload
    assert create_payload(SCENARIOS['bearing'].state, 'kev-alias', True)['model'] == 'kev-alias'
    assert list(create_payload(SCENARIOS['bearing'].state, only_fault=True)['questions']) == ['fault']


def test_parse_success():
    result = parse_decision(sample_reply())
    assert result.choice == 'bearing_wear'
    assert result.severity_score == pytest.approx(2.2)
    assert result.inspect_probability == pytest.approx(0.92)
    assert result.input_tokens == 235


def test_parse_choice_only():
    response = sample_reply()
    response['answers'] = {'fault':response['answers']['fault']}
    result = parse_decision(response, only_fault=True)
    assert result.severity_score is None
    assert result.inspect_probability is None


def test_bad_answer_cannot_masquerade_as_live_inference():
    response = sample_reply()
    del response['answers']['fault']['probabilities']['normal']
    with pytest.raises(DecisionModelError):
        parse_decision(response)


def test_localhost_only():
    assert local_server_url('http://localhost:8080/') == 'http://localhost:8080'
    assert local_server_url('http://127.0.0.1:8080') == 'http://127.0.0.1:8080'
    for host in ['https://localhost:8080', 'http://remote.example.com:8080',
                 'http://127.0.0.1:8080/anywhere', 'http://localhost:8080?a=1', 'http://localhost:invalid']:
        with pytest.raises(DecisionModelError):
            local_server_url(host)


def test_mocked_http_inference_calls_real_endpoint_contract():
    response = Mock()
    response.json.return_value = sample_reply()
    response.raise_for_status.return_value = None
    with patch('model_client.requests.post',return_value=response) as mock_post:
        decision = run_inference(SCENARIOS['bearing'].state)
        assert decision.choice == 'bearing_wear'
        args, kwargs = mock_post.call_args
        assert args[0] == 'http://127.0.0.1:8080/v1/systemone'
        assert kwargs['json']['questions']['fault']['type'] == 'choice'


def test_server_404_has_useful_error():
    response = Mock(status_code=404, text='Not found')
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    with patch('model_client.requests.post', return_value=response):
        with pytest.raises(DecisionModelError, match='Update llama.cpp'):
            run_inference(SCENARIOS['bearing'].state)


def test_probe_disconnected_is_not_a_fake_success():
    with patch('model_client.requests.get',side_effect=requests.ConnectionError('offline')):
        assert probe_server('http://127.0.0.1:8080')['healthy'] is False
