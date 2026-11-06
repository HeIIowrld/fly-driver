import json
import pytest
from flybrain_drive.graph import sha256
from scripts.audit_initial_thermal_evidence import initialization_receipts


def write(path,value):path.write_text(json.dumps(value),encoding='utf8')


def fixture(tmp_path):
    original=tmp_path/'failed-init';original.mkdir();verify=tmp_path/'read-only';verify.mkdir()
    request={'id':'one','temperature_c':80};terminal={'id':'one','status':'thermal_calls_returned_readback_required'};publication={'host_clock':1.}
    for name,value in [('request',request),('api-receipt',terminal),('publication',publication),('result',{'status':'failed'})]:write(original/(name+'.json'),value)
    write(verify/'manifest.json',{'config':{'arguments':{'initialization_run':str(original)}}})
    write(verify/'original-receipt.json',dict(request=request,terminal=terminal,publication=publication,
        original_request_sha256=sha256(original/'request.json'),original_terminal_sha256=sha256(original/'api-receipt.json')))
    return original,verify


def test_read_only_receipts_resolve_without_reclassifying_failed_original(tmp_path):
    original,verify=fixture(tmp_path)
    request,terminal,publication,resolved=initialization_receipts(verify,{'readback_only':True})
    assert resolved==original and request['id']==terminal['id']=='one'
    assert json.loads((original/'result.json').read_text())['status']=='failed'


@pytest.mark.parametrize('filename',['request.json','api-receipt.json','publication.json'])
def test_altered_original_is_not_silently_replaced_by_embedded_copy(tmp_path,filename):
    original,verify=fixture(tmp_path);write(original/filename,{'tampered':True})
    with pytest.raises(ValueError):initialization_receipts(verify,{'readback_only':True})
