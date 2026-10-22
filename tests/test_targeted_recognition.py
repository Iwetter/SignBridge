"""Targeted diagnostics/features use numeric fixtures, not live ASL evidence."""
import json,os
from pathlib import Path
import numpy as np
import pytest
from src.contracts import Prediction,STATIC_LABELS
from src.recognition import CPURecognizer
from src.recognition.geometric import RAW,GEOMETRIC,model_features,feature_names
from src.recognition.dataset import probability_matrix,load_dataset
from src.recognition.diagnostics import prediction_diagnostics
from src.temporal import TemporalDecoder
from src.vision.features import normalize_landmarks
from training import train,evaluate
from training.export_quantized import export_model


def numeric_samples(path):
    path.mkdir()
    rng=np.random.default_rng(73)
    for session in ('one','two'):
        for label,index in [('N',3),('T',7)]:
            x=rng.normal(0,.01,(50,63)).astype(np.float32);x[:,index]+=1
            seq=np.zeros((50,134),np.float32);seq[:,132]=1
            np.savez_compressed(path/f'{label}-{session}.npz',features=x,sequence=seq,
                label=np.asarray(label),session_id=np.asarray(session),schema_version=np.asarray(1),
                timestamps_ms=np.arange(50,dtype=np.int64)*100)


def test_geometry_shared_batch_single_names_and_orientation_preservation():
    rng=np.random.default_rng(19)
    raw=np.stack([normalize_landmarks(rng.random((21,3))) for _ in range(5)])
    x=model_features(raw,GEOMETRIC)
    assert x.shape==(5,len(feature_names(GEOMETRIC))) and x.dtype==np.float32
    np.testing.assert_array_equal(x[:,:63],raw)
    for i in range(5):np.testing.assert_allclose(x[i],model_features(raw[i],GEOMETRIC),atol=1e-6)
    rotated=raw[0].reshape(21,3)@np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32)
    assert not np.allclose(model_features(rotated.reshape(63),GEOMETRIC),x[0])
    assert np.isfinite(model_features(np.zeros(63),GEOMETRIC)).all()
    for bad in (np.full(63,np.nan),np.zeros(62)):
        with pytest.raises(ValueError):model_features(bad,GEOMETRIC)


def test_geometric_training_reload_evaluation_and_export_guard(tmp_path):
    data=tmp_path/'samples';numeric_samples(data);model_path=tmp_path/'candidate.npz'
    assert train.main(['--data-dir',str(data),'--output',str(model_path),'--ready-only',
        '--feature-set',GEOMETRIC,'--calibrate-temperature','--epochs','35','--data-origin','synthetic_fixture'])==0
    model=CPURecognizer(model_path);dataset=load_dataset(data)
    expected=probability_matrix(model_features(dataset.features,GEOMETRIC),model.arrays,list(model.supported_labels),model.temperature)
    actual=np.stack([model.predict(x,i).scores for i,x in enumerate(dataset.features)])
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    again=CPURecognizer(model_path)
    np.testing.assert_array_equal(model.predict(dataset.features[0],0).scores,again.predict(dataset.features[0],0).scores)
    assert model.metadata['feature_names']==feature_names(GEOMETRIC)
    report=tmp_path/'evaluation.json'
    assert evaluate.main(['--data-dir',str(data),'--model',str(model_path),'--output',str(report),'--benchmark-samples','3'])==0
    assert json.loads(report.read_text())['evaluation']['samples']==100
    with pytest.raises(ValueError,match='CPU-only'):export_model(model_path,tmp_path/'bad.npz',tmp_path/'rom')
    assert not (tmp_path/'bad.npz').exists()
    with np.load(model_path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    meta=json.loads(str(arrays['metadata'].item()));meta['feature_names']=[];arrays['metadata']=np.asarray(json.dumps(meta))
    np.savez_compressed(tmp_path/'bad-schema.npz',**arrays)
    with pytest.raises(ValueError,match='feature names'):CPURecognizer(tmp_path/'bad-schema.npz')


def test_calibration_is_global_preserves_argmax_and_validates_temperature():
    from src.recognition.cpu import masked_softmax
    logits=np.arange(26,dtype=np.float32)/10
    old=masked_softmax(logits,list(STATIC_LABELS));new=masked_softmax(logits,list(STATIC_LABELS),.7)
    assert old.argmax()==new.argmax() and new[9]==new[25]==0
    assert np.argsort(old).tolist()==np.argsort(new).tolist()
    for invalid in (0,-1,np.nan,np.inf):
        with pytest.raises(ValueError):masked_softmax(logits,list(STATIC_LABELS),invalid)


def test_diagnostics_separates_classifier_rejection_dwell_and_repeat_lock():
    decoder=TemporalDecoder();scores=np.zeros(26,np.float32);scores[13]=.79;scores[19]=.21
    p=Prediction('N',.79,0,scores=scores)
    low=prediction_diagnostics(p,decoder.snapshot(),['N','T'])
    assert low['classifier_candidate']=='N' and low['top_five'][0]['label']=='N'
    assert 'below confidence' in low['classifier_decision'] and 'classifier confidence rejected' in low['temporal_decision']
    p.confidence=.95;scores[13]=.95;scores[19]=.05;decoder.update(p,0)
    assert 'Waiting for stable' in prediction_diagnostics(p,decoder.snapshot(),['N','T'])['temporal_decision']
    for t in (100,200,300,400):p.timestamp_ms=t;decoder.update(p,t)
    assert 'held-letter' in prediction_diagnostics(p,decoder.snapshot(),['N','T'])['temporal_decision']
    assert prediction_diagnostics(p,decoder.snapshot(),['N','T'],fresh=False)['top_five']==[]
    assert prediction_diagnostics(p,decoder.snapshot(),['N','T'],replay=True)['top_five']==[]


def test_focused_calibration_navigation_and_busy_state():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from src.ui.desktop import SignBridgeUI
    ui=SignBridgeUI(lambda *a:None)
    try:
        for name,letters in [('T / N',['T','N']),('U / R / V',['U','R','V']),('S / E',['S','E']),('Q / H',['Q','H'])]:
            ui.calibration_group.setCurrentText(name)
            assert [ui.labels.itemText(i) for i in range(ui.labels.count())]==letters
            ui.labels.setCurrentText(letters[-1]);ui.next_label.click();assert ui.labels.currentText()==letters[0]
        ui.update(dict(recording=True,recording_stats={'label':'Q'},source='live'));ui._drain()
        assert not ui.calibration_group.isEnabled() and not ui.labels.isEnabled()
        ui.update(dict(source='live'));ui._drain();ui.calibration_group.setCurrentText('All static letters')
        assert tuple(ui.labels.itemText(i) for i in range(ui.labels.count()))==STATIC_LABELS
    finally:ui.window.close()


def test_focused_save_advances_ready_letter_only_after_persisted_receipt():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from src.ui.desktop import SignBridgeUI
    ui=SignBridgeUI(lambda *a:None)
    try:
        ui.calibration_group.setCurrentText('T / N');ui.auto_advance.setChecked(True)
        ui._static_progress={'T':{'ready':True},'N':{'ready':True}}
        ui._advance_after_save({'recording':True,'source':'live'},{},{})
        receipt=dict(saved_path='T-new.npz',saved_label='T',saved_samples=50,saved_session_id='new')
        ui._advance_after_save({'source':'live'},{},receipt)
        assert ui.labels.currentText()=='T'
        dataset={'files':[dict(path='T-new.npz',label='T',samples=50,session_id='new')]}
        ui._advance_after_save({'source':'live'},dataset,receipt)
        assert ui.labels.currentText()=='N'
        ui._advance_after_save({'source':'live'},dataset,receipt)
        assert ui.labels.currentText()=='N'
    finally:ui.window.close()


def test_promotion_policy_rejects_confidence_regression_even_when_argmax_improves():
    from copy import deepcopy
    from scripts.targeted_recognition_audit import validation_checks
    old={c:dict(accuracy=.97,confident_correct_fraction=.90,confident_error_fraction=.01) for c in STATIC_LABELS}
    new=deepcopy(old)
    assert all(validation_checks(old,new,['A','C','N']).values())
    new['C'].update(accuracy=.99,confident_correct_fraction=.80)
    checks=validation_checks(old,new,['A','C','N'])
    assert checks['C_regression'] and not checks['C_confident_correct_regression']
    new['C']=deepcopy(old['C']);new['A']['confident_error_fraction']=.08
    assert not validation_checks(old,new,['A','C','N'])['A_confident_error_regression']


def test_calibrated_raw_model_export_fails_before_writing(tmp_path):
    data=tmp_path/'samples';numeric_samples(data);path=tmp_path/'model.npz'
    assert train.main(['--data-dir',str(data),'--output',str(path),'--epochs','2','--data-origin','synthetic_fixture'])==0
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    metadata=json.loads(str(arrays['metadata'].item()));metadata['temperature']=.7
    arrays['metadata']=np.asarray(json.dumps(metadata));np.savez_compressed(path,**arrays)
    CPURecognizer(path)
    with pytest.raises(ValueError,match='temperature'):export_model(path,tmp_path/'quantized.npz',tmp_path/'rom')
    assert not (tmp_path/'quantized.npz').exists() and not (tmp_path/'rom').exists()
