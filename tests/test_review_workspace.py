"""AI display cannot silently become human labels or verified asset identities."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


_path = Path(__file__).resolve().parents[1] / "scripts/build_review_workspace.py"
_spec = importlib.util.spec_from_file_location("review_workspace", _path)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def _row(**changes):
    return {"id": "sentfin:7:Acme", "text": "Acme shares rally after forecasts rise.",
            "source_entity": "Acme", "ai_stance": "bullish", "ai_rationale": "盈利前景改善。",
            "evidence_quotes": ["forecasts rise"], "human_reviewed": False,
            "source_symbol_candidates": ["OLD_ACME"], **changes}


def _input(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def _core():
    return re.search(r'<script id="review-core">(.*?)</script>', review.TEMPLATE, re.S).group(1)


def _node(code):
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node is required for actual JavaScript decision-core checks")
    result = subprocess.run([executable, "-e", code], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


def test_workspace_embeds_literal_data_without_executable_html_or_approval(tmp_path):
    hostile = '</script><img src=x onerror="alert(1)"> & quoted "headline"'
    source = _input(tmp_path, [_row(text=hostile)])
    output = tmp_path / "review.html"
    metadata = review.build(source, output)
    html = output.read_text(encoding="utf-8")
    assert metadata["rows"] == 1 and metadata["human_approvals_created"] == 0
    assert hostile not in html
    encoded = re.search(r'<script type="application/json" id="review-data">(.*?)</script>', html, re.S).group(1)
    decoded = json.loads(encoded)
    assert decoded["rows"][0]["text"] == hostile
    assert decoded["rows"][0]["human_reviewed"] is False
    assert len(re.findall(r"<script\b", html)) == 3
    assert "connect-src 'none'" in html
    assert review.build(source, output) == metadata


@pytest.mark.parametrize("rows", [
    [_row(), _row()], [_row(human_reviewed=True)], [_row(ai_stance="positive")],
    [_row(text="")], [_row(source_entity=17)],
    [_row(secondary_ai_stance="positive", ai_disagreement=True)],
    [_row(secondary_ai_stance="bearish", ai_disagreement=False)],
])
def test_invalid_ai_input_fails_before_writing_workspace(tmp_path, rows):
    source = _input(tmp_path, rows)
    output = tmp_path / "workspace.html"
    with pytest.raises(ValueError):
        review.build(source, output)
    assert not output.exists()


def test_changed_workspace_is_not_overwritten(tmp_path):
    source = _input(tmp_path, [_row()])
    output = tmp_path / "review.html"
    review.build(source, output)
    output.write_text("human local edit", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing workspace differs"):
        review.build(source, output)
    assert output.read_text() == "human local edit"


def test_actual_js_import_binds_dataset_ids_and_distinguishes_asset_approval():
    _node(_core() + r'''
const assert=require('node:assert/strict');
const d={id:'r1',human_stance:'bullish',asset_identifier:'OLD_ACME',asset_verified:false,
 notes:'manual label only',event_group_id:'',human_reviewed:true,review_status:'reviewed',reviewed_at:'2026-10-30T00:00:00Z'};
const backup={format:'130u-review-decisions-v1',dataset_sha256:'frozen-sha',row_ids:['r1','r2'],decisions:[d]};
const imported=ReviewCore.validateImport(backup,'frozen-sha',['r1','r2']);
assert.equal(imported.r1.human_reviewed,true);
assert.equal(imported.r1.asset_verified,false);
assert.throws(()=>ReviewCore.validateImport(backup,'other-sha',['r1','r2']));
assert.throws(()=>ReviewCore.validateImport(backup,'frozen-sha',['r2','r1']));
assert.throws(()=>ReviewCore.validateImport({...backup,decisions:[d,d]},'frozen-sha',['r1','r2']));
assert.throws(()=>ReviewCore.validateImport({...backup,decisions:[{...d,id:'unknown'}]},'frozen-sha',['r1','r2']));
assert.throws(()=>ReviewCore.cleanDecision({...d,human_stance:'positive'},'r1'));
assert.throws(()=>ReviewCore.cleanDecision({...d,asset_verified:true,asset_identifier:''},'r1'));
assert.throws(()=>ReviewCore.cleanDecision({...d,review_status:'draft'},'r1'));
assert.equal(ReviewCore.cleanDecision({...d,human_stance:'exclude',asset_identifier:''},'r1').human_reviewed,true);
''')


def test_actual_js_csv_exports_only_reviewed_rows_and_separates_unverified_asset_draft():
    _node(_core() + r'''
const assert=require('node:assert/strict');
const base={id:'r1',human_stance:'bullish',asset_identifier:'OLD_ACME',asset_verified:false,
 notes:'=HYPERLINK("bad")',event_group_id:'',human_reviewed:true,review_status:'reviewed',reviewed_at:'now'};
const csv=ReviewCore.csv([base,{...base,id:'unreviewed-id',human_reviewed:false,review_status:'draft'}]);
assert.equal(csv.charCodeAt(0),0xfeff);
assert.ok(csv.includes('"r1","","OLD_ACME"'));
assert.ok(!csv.includes('unreviewed-id'));
assert.ok(csv.includes("'=HYPERLINK"));
assert.ok(ReviewCore.csv([{...base,asset_verified:true}]).includes('"r1","OLD_ACME",""'));
for(const text of ['=1+1','+cmd','-cmd','@SUM(1)','  =1','\t=1']) {
 assert.ok(ReviewCore.csvCell(text).startsWith('"\''));
}
''')


def test_all_executable_javascript_parses_and_blind_mode_is_not_ai_prefilled():
    scripts = re.findall(r'<script(?: id="review-core")?>(.*?)</script>', review.TEMPLATE, re.S)
    assert len(scripts) == 2
    _node("const vm=require('node:vm');\n" + "\n".join("new vm.Script(" + json.dumps(s) + ");" for s in scripts))
    assert "d.human_stance||($('blindMode').checked?'':r.ai_stance)" in review.TEMPLATE
    assert "$('labelFilter').disabled=$('blindMode').checked" in review.TEMPLATE
    assert "formal_training_ready:false" in review.TEMPLATE
    assert "const confirmed=ReviewCore.cleanDecision(d,r.id)" in review.TEMPLATE


def test_actual_ui_events_require_explicit_approval_keep_asset_draft_and_preserve_failed_import():
    main = re.findall(r"<script>(.*?)</script>", review.TEMPLATE, re.S)[0]
    rows = [_row(id="r1"), _row(id="r2", ai_stance="neutral")]
    data = json.dumps({"dataset_sha256": "bound-sha", "rows": rows, "title": "test"}, ensure_ascii=False)
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.value='';this.checked=false;this.disabled=false;this.style={};this.listeners={};this.children=[];this.textContent='';this.files=[];}
 addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}
 replaceChildren(...values){this.children=[...values];}
 append(...values){this.children.push(...values);}
 click(){if(this.onclick)this.onclick();}
 remove(){}
 closest(){return null;}
}
const elements=new Map(),storage=new Map(),blobs=[],documentListeners={};
global.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
 createElement(){return new Element();},createTextNode(text){const e=new Element();e.textContent=text;return e;},
 body:new Element(),addEventListener(name,fn){(documentListeners[name]??=[]).push(fn);}};
document.body.classList={toggle(){}};
global.localStorage={getItem(k){return storage.get(k)??null;},setItem(k,v){storage.set(k,v);}};
global.URL={createObjectURL(blob){blobs.push(blob);return 'blob:test';},revokeObjectURL(){}};
global.setTimeout=fn=>fn();
for(const id of ['priorityFilter','difficultyFilter','labelFilter','batchFilter'])document.getElementById(id).value='all';
document.getElementById('statusFilter').value='pending';
'''
    events = r'''
(async()=>{
assert.equal(storage.size,0,'rendering cannot produce any human decision');
const blind=document.getElementById('blindMode');blind.checked=true;blind.listeners.change[0]();
assert.equal(document.getElementById('humanLabel').value,'','blind mode cannot prefill AI stance');
assert.equal(document.getElementById('labelFilter').disabled,true);
assert.equal(document.getElementById('labelFilter').value,'all');
document.getElementById('approveButton').click();
assert.equal(storage.size,0,'blank blind stance cannot be approved');
const keydown=documentListeners.keydown[0];
keydown({key:'3',target:{closest:()=>null},preventDefault(){}});
let draft=JSON.parse(storage.get('130u-finllm-review:bound-sha'));
assert.equal(draft.decisions[0].human_stance,'bullish');
assert.equal(draft.decisions[0].human_reviewed,false,'number key selects but never approves');
const beforeTyping=storage.get('130u-finllm-review:bound-sha');
keydown({key:'a',target:{closest:()=>({})},preventDefault(){throw Error('typing must not trigger shortcut');}});
for(const modifier of ['ctrlKey','altKey','metaKey'])keydown({key:'a',[modifier]:true,target:{closest:()=>null},preventDefault(){throw Error('modifier must not trigger approval');}});
keydown({key:'a',repeat:true,target:{closest:()=>null},preventDefault(){throw Error('holding A must not mass-approve');}});
assert.equal(storage.get('130u-finllm-review:bound-sha'),beforeTyping);
blind.checked=false;blind.listeners.change[0]();
document.getElementById('approveButton').click();
let saved=JSON.parse(storage.get('130u-finllm-review:bound-sha'));
assert.equal(saved.decisions.length,1);
assert.equal(saved.decisions[0].id,'r1');
assert.equal(saved.decisions[0].human_reviewed,true);
assert.equal(saved.decisions[0].asset_verified,false);
document.getElementById('undoButton').click();
let undone=JSON.parse(storage.get('130u-finllm-review:bound-sha'));
assert.equal(undone.decisions.find(d=>d.id==='r1').human_reviewed,false);
document.getElementById('approveButton').click();
document.getElementById('statusFilter').value='pending';document.getElementById('statusFilter').listeners.change[0]();
const asset=document.getElementById('humanAsset');asset.value='NASDAQ:UNVERIFIED';asset.listeners.input[0]();
const verification=document.getElementById('assetVerified');verification.checked=true;verification.listeners.change[0]();
asset.value='NASDAQ:CHANGED';asset.listeners.input[0]();
assert.equal(verification.checked,false,'changing the asset invalidates prior asset verification');
document.getElementById('approveButton').click();
document.getElementById('allJsonButton').click();
const exported=(await blobs.at(-1).text()).trim().split('\n'.replace('\\n','\n')).map(JSON.parse);
assert.equal(exported.length,2);
assert.equal(exported[1].human_reviewed,true);
assert.equal(exported[1].asset_verified,false);
assert.equal(exported[1].asset_identifier,'');
assert.equal(exported[1].asset_identifier_draft,'NASDAQ:CHANGED');
assert.equal(exported[1].formal_training_ready,false);
const filter=document.getElementById('statusFilter');filter.value='all';filter.listeners.change[0]();
const label=document.getElementById('humanLabel');label.value='bearish';label.listeners.input[0]();
saved=JSON.parse(storage.get('130u-finllm-review:bound-sha'));
assert.equal(saved.decisions.find(d=>d.id==='r1').human_reviewed,false,'editing revokes stale approval');
const before=storage.get('130u-finllm-review:bound-sha');
document.getElementById('importFile').files=[{text:async()=>JSON.stringify({...saved,dataset_sha256:'wrong'})}];
await document.getElementById('importFile').onchange();
assert.equal(storage.get('130u-finllm-review:bound-sha'),before);
assert.ok(document.getElementById('toast').textContent.includes('导入失败'));
global.localStorage.setItem=()=>{throw Error('storage denied');};
label.value='neutral';label.listeners.input[0]();
assert.equal(document.getElementById('storageWarning').style.display,'block');
document.getElementById('backupButton').click();
const backupAfterStorageFailure=JSON.parse(await blobs.at(-1).text());
assert.equal(backupAfterStorageFailure.decisions.find(d=>d.id==='r1').human_stance,'neutral');
assert.equal(backupAfterStorageFailure.decisions.find(d=>d.id==='r1').human_reviewed,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    # The real generated main script runs against a minimal DOM/localStorage
    # substitute. This tests event behavior, not browser layout or file permissions.
    _node(harness + "\ndocument.getElementById('review-data').textContent=" + json.dumps(data) + ";\n" + _core() + main + events)
