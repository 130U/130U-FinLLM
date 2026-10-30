"""Build a standalone, offline human-review workspace from AI-only JSONL.

The generated file contains local source text. Keep it outside public Git.
Viewing a suggestion never creates a human approval; decisions are explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


STANCE = {"bearish", "neutral", "bullish", "exclude"}


def load_rows(path):
    raw = Path(path).read_bytes()
    rows, ids = [], set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "text", "source_entity")):
            raise ValueError(f"Row {number}: id, text and source_entity must be nonempty strings")
        if row["id"] in ids:
            raise ValueError(f"Duplicate review id: {row['id']}")
        if row.get("human_reviewed", False) not in (False, "false", "", None):
            raise ValueError("AI workspace input must not assert human approval; import actual decisions through the UI")
        suggestion = row.get("ai_stance", row.get("proposed_stance", ""))
        if suggestion not in STANCE:
            raise ValueError(f"Row {number}: invalid AI stance suggestion")
        secondary = row.get("secondary_ai_stance")
        if secondary and (secondary not in STANCE or not isinstance(row.get("ai_disagreement"), bool)
                          or row["ai_disagreement"] != (secondary != suggestion)):
            raise ValueError(f"Row {number}: secondary AI label or disagreement flag is inconsistent")
        row = {**row, "ai_stance": suggestion}
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("Review input is empty")
    return rows, hashlib.sha256(raw).hexdigest()


def build(input_jsonl, output_html, title="130U-FinLLM｜资产立场审核工作台"):
    rows, digest = load_rows(input_jsonl)
    payload = json.dumps({"dataset_sha256": digest, "rows": rows, "title": title}, ensure_ascii=False)
    # JSON is data, never executable HTML. In particular, prevent </script>.
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = TEMPLATE.replace("__DATA_PAYLOAD__", payload)
    output_html = Path(output_html)
    if output_html.exists() and output_html.read_text(encoding="utf-8") != page:
        raise ValueError("Existing workspace differs; use a new output filename to preserve its review context")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")
    return {"rows": len(rows), "dataset_sha256": digest, "html_sha256": hashlib.sha256(page.encode()).hexdigest(),
            "human_approvals_created": 0}


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>130U-FinLLM 审核工作台</title>
<style>
:root{color-scheme:light;--bg:#f4f5f3;--paper:#fff;--ink:#192b28;--muted:#647570;--line:#d8e1dc;--accent:#176a58;--soft:#e9f4ee;--amber:#fff3da;--red:#a33f38;--shadow:0 8px 30px #203a2910}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif}button,input,select,textarea{font:inherit}button,select,input,textarea{border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--ink)}button{padding:8px 13px;cursor:pointer}button:hover{border-color:var(--accent);background:var(--soft)}button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}button.danger{color:var(--red)}button:disabled{opacity:.45;cursor:default}a{color:var(--accent)}header{background:var(--paper);border-bottom:1px solid var(--line);padding:20px 28px;display:flex;gap:25px;justify-content:space-between;align-items:center}.brand{font-size:20px;font-weight:750;letter-spacing:-.5px}.subtle{font-size:13px;color:var(--muted)}.pill{display:inline-block;padding:2px 9px;border-radius:20px;background:var(--soft);color:var(--accent);font-size:12px}.layout{max-width:1500px;margin:auto;display:grid;grid-template-columns:250px 1fr;gap:20px;padding:22px}.panel,.card{background:var(--paper);border:1px solid var(--line);border-radius:13px;box-shadow:var(--shadow)}.panel{padding:18px;height:max-content}.panel label{display:block;font-size:12px;color:var(--muted);margin:15px 0 5px}.panel select,.panel input{width:100%;padding:7px}.panel select:first-child{margin-top:0}main{min-width:0}.card{padding:26px;margin-bottom:16px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.between{justify-content:space-between}.caption{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:650}.target{font-size:24px;font-weight:750;margin:6px 0 12px}.headline{font-size:24px;line-height:1.55;margin:8px 0 22px;font-weight:550;white-space:pre-wrap}.ai{border-left:3px solid var(--accent);padding:13px 17px;background:var(--soft);border-radius:0 9px 9px 0}.ai p{margin:4px 0}.evidence{font-family:Georgia,serif;color:var(--muted);font-size:16px}.label{display:block;font-size:13px;font-weight:650;margin:15px 0 5px}input[type=text],textarea,.review-label{width:100%;padding:9px 11px}textarea{height:76px;resize:vertical}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.check{font-size:13px;display:flex;align-items:flex-start;gap:8px;margin-top:9px}.check input{margin-top:6px}.actions{margin-top:23px;padding-top:18px;border-top:1px solid var(--line)}.actions .primary{font-weight:650}.hint{font-size:12px;color:var(--muted);margin-top:10px}details{border-top:1px solid var(--line);margin-top:16px;padding-top:12px}summary{cursor:pointer;color:var(--muted);font-size:13px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}.progress{height:7px;background:var(--line);border-radius:8px;margin:10px 0}.progress span{height:100%;display:block;background:var(--accent);border-radius:8px}#toast{min-height:24px;color:var(--accent);font-size:13px;margin:8px 0}#storageWarning{display:none;background:var(--amber);padding:10px 16px;font-size:13px}#empty{display:none}.instructions{font-size:13px}.instructions p{margin:7px 0}.kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:11px}.blind .ai{display:none}.blind .source-sentiment{display:none}.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin-top:14px}.toolbar button{font-size:12px;padding:6px 9px}.metric{font-size:28px;font-weight:750;letter-spacing:-1px}.row-id{font-size:12px;color:var(--muted);word-break:break-word}.batch-card{margin-top:20px;border-top:1px solid var(--line);padding-top:12px}#batchProgress{font-size:12px;max-height:250px;overflow:auto}.batch-line{display:flex;justify-content:space-between;padding:4px 0}.tag{background:var(--bg);border:1px solid var(--line);border-radius:5px;padding:2px 6px;font-size:12px}.status{font-weight:650}.warm{background:var(--amber);border-left-color:#d29a35}.light-note{margin-bottom:18px;border:1px solid var(--line);border-radius:8px;padding:10px 14px;background:var(--bg);font-size:13px}input[type=checkbox]{accent-color:var(--accent)}mark{background:var(--amber);color:inherit;padding:0 3px;border-radius:3px}.tag.high,.tag.hard{background:var(--amber)}@media(max-width:900px){header{padding:15px;align-items:flex-start}.layout{padding:12px;grid-template-columns:1fr}.panel{display:grid;grid-template-columns:1fr 1fr;gap:8px}.panel .full{grid-column:1/-1}.panel label{margin-top:4px}.card{padding:18px}.headline{font-size:20px}.form-grid{grid-template-columns:1fr}.brand{font-size:17px}}@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#14201d;--paper:#1c2b26;--ink:#e9f1eb;--muted:#a0b3a9;--line:#35473f;--accent:#78c7a5;--soft:#233e32;--amber:#4b3d1f;--red:#f39b94;--shadow:none}button.primary{background:#2d8061;color:#fff;border-color:#2d8061}}
</style></head><body>
<header><div><div class="brand" id="workspaceTitle"></div><div class="subtle">英文原文 · 指定目标 · AI 初判 · 逐条人工确认</div></div><div><span class="pill">本地离线工作台</span><div class="subtle">不上传数据，不自动批准标签</div></div></header>
<div id="storageWarning">浏览器未允许本地自动保存。请经常导出「进度 JSON」，关闭前务必备份；以后用「导入进度」恢复。</div>
<div class="layout"><aside class="panel">
<div class="full"><div class="caption">人工立场审核进度</div><div><span class="metric" id="doneCount">0</span><span id="totalCount" class="subtle"></span></div><div class="progress"><span id="progressBar"></span></div><div id="progressSummary" class="subtle"></div></div>
<div><label for="statusFilter">审核状态</label><select id="statusFilter"><option value="pending">待审立场</option><option value="all">全部</option><option value="reviewed">立场已确认</option><option value="asset_pending">立场已审／资产待核实</option><option value="asset_verified">立场／资产双确认</option><option value="uncertain">待查／存疑</option><option value="draft">已修改未确认</option></select></div>
<div><label for="priorityFilter">优先级</label><select id="priorityFilter"><option value="all">全部优先级</option></select></div>
<div><label for="difficultyFilter">难度</label><select id="difficultyFilter"><option value="all">全部难度</option></select></div>
<div><label for="labelFilter">AI 建议</label><select id="labelFilter"><option value="all">全部类别</option><option value="bearish">看空</option><option value="neutral">中性</option><option value="bullish">看多</option><option value="exclude">排除</option></select></div>
<div><label for="disagreementFilter">二次 AI 复核</label><select id="disagreementFilter"><option value="all">全部</option><option value="disagree">两次 AI 建议不同</option><option value="audited">已有二次 AI 建议</option><option value="not_audited">未做二次 AI 复核</option></select></div>
<div><label for="batchFilter">审核批次</label><select id="batchFilter"><option value="all">全部批次</option></select></div>
<div><label for="groupFilter">原文／事件组</label><input id="groupFilter" placeholder="输入组号筛选"></div>
<div class="full"><label for="searchFilter">检索原文、目标或行号</label><input id="searchFilter" placeholder="例如 earnings / Apple"></div>
<div class="full"><label class="check"><input id="calibrationMode" type="checkbox">先审规则校准 40 条（不是封存测试）</label></div>
<div class="full"><label class="check"><input id="blindMode" type="checkbox">盲审模式：先隐藏 AI 建议和来源情绪标签</label></div>
<div class="full toolbar"><button id="undoButton">撤回上一操作 Z</button><button id="backupButton">备份进度 JSON</button><button id="importButton">导入进度</button><button id="reviewedCsvButton">已确认 CSV</button><button id="reviewedJsonButton">已确认 JSONL</button><button id="allJsonButton">全部行 JSONL</button><input type="file" id="importFile" accept="application/json,.json" hidden></div>
<div class="full subtle">本地缓存不是文件备份。每完成一批，请导出进度 JSON；CSV 为 Excel 转义公式字符，原始文本与训练导入请使用 JSONL。</div>
<div class="full batch-card"><div class="caption">各批次进度</div><div id="batchProgress"></div></div>
</aside><main><section class="card instructions"><div class="row between"><strong>每条只做三件事</strong><span class="subtle" id="datasetShort"></span></div><p>① 先读英文原文与指定目标　② 核对 AI 判断和资产身份　③ 明确确认或修改后确认。</p><p>只描述过去涨跌／财报数字，通常不足以单独推出投资方向；按原文表达的观点判断。无法确认目标、语气或资产身份时标「待查」或排除。</p><p>快捷键：<span class="kbd">A</span> 确认并下一条　<span class="kbd">1</span> 看空　<span class="kbd">2</span> 中性　<span class="kbd">3</span> 看多　<span class="kbd">X</span> 排除并确认　<span class="kbd">U</span> 待查　<span class="kbd">J / K</span> 下一条／上一条　<span class="kbd">Z</span> 撤回上一确认／待查。输入框内不触发快捷键。</p></section>
<div id="toast" role="status" aria-live="polite"></div>
<section id="empty" class="card"><h2>当前筛选下没有条目</h2><p>可以切换到「全部」，或调整类别／批次筛选。</p></section>
<section id="reviewCard" class="card"><div class="row between"><div id="rowMeta" class="row"></div><div class="row"><button id="previousButton">← 上一条</button><span id="position" class="subtle"></span><button id="nextButton">下一条 →</button></div></div>
<div class="caption" style="margin-top:22px">本条指定目标</div><div id="target" class="target"></div><div id="rowId" class="row-id"></div><div id="headline" class="headline" lang="en"></div>
<div id="groupNote" class="light-note"></div>
<div class="ai"><div class="row between"><strong id="aiLabel"></strong><span id="aiUncertainty" class="subtle"></span></div><p id="aiRationale"></p><p id="aiEvidence" class="evidence"></p><div class="subtle" id="assetSuggestion"></div><div id="secondaryAi" class="light-note" style="display:none;margin:12px 0 0"><strong id="secondaryLabel"></strong><p id="secondaryReason"></p><div class="subtle">这是第二次 AI 建议，仍待人工审核；两次一致也不代表标签正确。</div></div></div>
<div class="form-grid"><div><label class="label" for="humanLabel">你的最终立场</label><select id="humanLabel" class="review-label"><option value="">请选择</option><option value="bearish">bearish｜看空</option><option value="neutral">neutral｜中性</option><option value="bullish">bullish｜看多</option><option value="exclude">exclude｜排除</option></select></div><div><label class="label" for="humanAsset">已核实的资产标识</label><input id="humanAsset" type="text" placeholder="核实后填，不要直接照抄历史代码"><label class="check"><input type="checkbox" id="assetVerified">我已核实这条指定目标的资产身份（排除项不要求）</label></div></div>
<div class="form-grid"><div><label class="label" for="humanNotes">审核备注／修正依据</label><textarea id="humanNotes" placeholder="可留空；修正或存疑时建议写原因"></textarea></div><div><label class="label" for="humanEvent">人工事件组（可暂留空）</label><input id="humanEvent" type="text" placeholder="同一事件填写同一组号"><div class="hint">自动分组只辅助排序，不能替代语义／事件重复审核。</div><div class="hint status" id="currentStatus"></div></div></div>
<div class="row actions"><button class="primary" id="approveButton">确认并下一条 A</button><button id="uncertainButton">标为待查 U</button><button class="danger" id="excludeButton">排除并确认 X</button><button id="resetButton">撤回本条确认</button></div>
<div class="hint">确认按钮只批准本条立场。资产需另勾选核实；没有勾选的资产标识只作为草稿导出。AI 建议、查看和暂存都不算人工批准；没有「全部批准」操作。完成此表后仍需分组、权限和数据切分验收。</div>
<details><summary>查看来源与 AI 原始记录</summary><div class="source-sentiment"><div id="sourceLabel" class="hint"></div></div><pre id="sourceDetails"></pre></details>
</section></main></div>
<script type="application/json" id="review-data">__DATA_PAYLOAD__</script>
<script id="review-core">
/* Pure decision logic is also exercised by the generator's Node tests. */
const ReviewCore=(()=>{
const STANCES=new Set(['bearish','neutral','bullish','exclude']);
const cleanDecision=(d,id)=>{
 if(!d||typeof d!=='object'||d.id!==id||!STANCES.has(d.human_stance)&&d.human_stance!=='')throw Error('审核记录的行号或标签无效');
 for(const k of ['asset_identifier','notes','event_group_id'])if(typeof d[k]!=='string')throw Error('审核记录字段必须是文本');
 if(typeof d.human_reviewed!=='boolean'||typeof d.asset_verified!=='boolean')throw Error('审核状态必须为布尔值');
 if(d.human_reviewed&&!STANCES.has(d.human_stance))throw Error('已确认条目缺少有效标签');
 if(d.asset_verified&&!d.asset_identifier.trim())throw Error('已核实资产缺少实际标识');
 if(!['draft','uncertain','reviewed'].includes(d.review_status)||d.human_reviewed!==(d.review_status==='reviewed'))throw Error('审核状态与确认标记不一致');
 return {id,human_stance:d.human_stance,asset_identifier:d.asset_identifier,asset_verified:d.asset_verified,notes:d.notes,event_group_id:d.event_group_id,human_reviewed:d.human_reviewed,review_status:d.review_status,reviewed_at:typeof d.reviewed_at==='string'?d.reviewed_at:''};
};
function validateImport(data,sha,ids){
 if(!data||data.format!=='130u-review-decisions-v1'||data.dataset_sha256!==sha||!Array.isArray(data.decisions))throw Error('这份进度不属于当前数据版本');
 if(!Array.isArray(data.row_ids)||data.row_ids.length!==ids.length||data.row_ids.some((id,i)=>id!==ids[i]))throw Error('进度中的原始条目顺序或身份不同');
 const result=Object.create(null),allowed=new Set(ids);
 for(const d of data.decisions){if(!allowed.has(d.id)||Object.hasOwn(result,d.id))throw Error('进度含未知或重复行号');result[d.id]=cleanDecision(d,d.id);}
 return result;
}
function csvCell(v){let s=String(v??'');if(/^[\s]*[=+@-]/.test(s)||/^[\t\r\n]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}
function csv(decisions){const fields=['id','asset_identifier','asset_identifier_draft','human_stance','human_reviewed','asset_verified','notes','event_group_id','reviewed_at'];return '\ufeff'+[fields.map(csvCell).join(','),...decisions.filter(d=>d.human_reviewed).map(d=>{const row={...d,asset_identifier:d.asset_verified?d.asset_identifier:'',asset_identifier_draft:d.asset_verified?'':d.asset_identifier};return fields.map(k=>csvCell(row[k])).join(',');})].join('\r\n');}
return {cleanDecision,validateImport,csvCell,csv};
})();
</script>
<script>
(()=>{'use strict';
const DATA=JSON.parse(document.getElementById('review-data').textContent),ROWS=DATA.rows,SHA=DATA.dataset_sha256,KEY='130u-finllm-review:'+SHA,IDS=ROWS.map(r=>r.id),BYID=new Map(ROWS.map(r=>[r.id,r]));
const $=id=>document.getElementById(id),LABELS={bearish:'看空 bearish',neutral:'中性 neutral',bullish:'看多 bullish',exclude:'排除 exclude'},QUEUE_NAMES={easy:'简单',medium:'中等',hard:'边界／高难',low:'常规',high:'优先复核'};
let decisions=Object.create(null),currentId=IDS[0],filtered=[],storeOk=true,decisionHistory=[];
const text=v=>Array.isArray(v)?v.join('；'):typeof v==='object'&&v?JSON.stringify(v):String(v??'');
const field=(r,...keys)=>{for(const k of keys)if(r[k]!==undefined&&r[k]!==null&&r[k]!=='')return r[k];return '';};
const group=r=>text(field(r,'group_id','text_group_id','source_headline_id'));
const batch=r=>text(field(r,'batch_id','batch'))||'未分批';
const CALIBRATION=(()=>{const result=new Set(),groups=new Set(),rank={easy:0,medium:1,hard:2};const ordered=[...ROWS].sort((a,b)=>(rank[a.difficulty]??1)-(rank[b.difficulty]??1));function add(r){const g=group(r)||r.id;if(result.has(r.id)||groups.has(g))return false;result.add(r.id);groups.add(g);return true;}for(const label of ['bearish','neutral','bullish','exclude']){let n=0;for(const r of ordered)if(r.ai_stance===label&&add(r)&&++n===10)break;}for(const r of ordered){if(result.size>=40)break;add(r);}return result;})();
function status(id){return decisions[id]?.review_status||'unreviewed';}
function envelope(){return {format:'130u-review-decisions-v1',dataset_sha256:SHA,row_ids:IDS,exported_at:new Date().toISOString(),formal_training_ready:false,decisions:Object.values(decisions)};}
function toast(message,error=false){$('toast').textContent=message;$('toast').style.color=error?'var(--red)':'var(--accent)';}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(envelope()));}catch(e){storeOk=false;$('storageWarning').style.display='block';}}
try{const saved=localStorage.getItem(KEY);if(saved)decisions=ReviewCore.validateImport(JSON.parse(saved),SHA,IDS);}catch(e){storeOk=false;$('storageWarning').style.display='block';toast('本地缓存无法使用，未加载任何批准。请导入有效的进度备份。',true);}
function renderHeadline(value,entity){const container=$('headline');container.replaceChildren();let offset=0,index;const lower=value.toLocaleLowerCase(),needle=entity.toLocaleLowerCase();while(needle&&(index=lower.indexOf(needle,offset))!==-1){container.append(document.createTextNode(value.slice(offset,index)));const mark=document.createElement('mark');mark.textContent=value.slice(index,index+entity.length);container.append(mark);offset=index+entity.length;}container.append(document.createTextNode(value.slice(offset)));}
function rowState(r){return decisions[r.id]||{id:r.id,human_stance:'',asset_identifier:'',asset_verified:false,notes:'',event_group_id:'',human_reviewed:false,review_status:'draft',reviewed_at:''};}
function update(fieldName,value){const r=BYID.get(currentId);if(!r)return;const old=rowState(r);if(old[fieldName]===value)return;decisions[r.id]={...old,[fieldName]:value,human_reviewed:false,review_status:'draft',reviewed_at:''};if(fieldName==='asset_identifier'){decisions[r.id].asset_verified=false;$('assetVerified').checked=false;}persist();updateProgress();$('currentStatus').textContent='已暂存修改，尚未确认';}
function updateProgress(){$('undoButton').disabled=decisionHistory.length===0;const done=Object.values(decisions).filter(d=>d.human_reviewed).length,dual=Object.values(decisions).filter(d=>d.human_reviewed&&d.asset_verified&&d.human_stance!=='exclude').length,uncertain=Object.values(decisions).filter(d=>d.review_status==='uncertain').length;$('doneCount').textContent=done;$('totalCount').textContent=' / '+ROWS.length;$('progressBar').style.width=(100*done/ROWS.length)+'%';$('progressSummary').textContent='立场已审 '+done+' · 双确认 '+dual+' · 待查 '+uncertain;const stats=new Map();for(const r of ROWS){const b=batch(r),s=stats.get(b)||[0,0];s[1]++;if(decisions[r.id]?.human_reviewed)s[0]++;stats.set(b,s);}$('batchProgress').replaceChildren();for(const [b,s]of stats){const line=document.createElement('div');line.className='batch-line';const a=document.createElement('span'),v=document.createElement('span');a.textContent=b;v.textContent=s[0]+'/'+s[1];line.append(a,v);$('batchProgress').append(line);}}
function matches(r){const s=$('statusFilter').value,rs=status(r.id),d=decisions[r.id];if($('calibrationMode').checked&&!CALIBRATION.has(r.id))return false;const audit=$('disagreementFilter').value,audited=!!r.secondary_ai_stance;if(audit==='disagree'&&r.ai_disagreement!==true||audit==='audited'&&!audited||audit==='not_audited'&&audited)return false;if(s==='asset_pending'&&(!d?.human_reviewed||d.human_stance==='exclude'||d.asset_verified))return false;if(s==='asset_verified'&&(!d?.human_reviewed||!d.asset_verified||d.human_stance==='exclude'))return false;if(s==='pending'&&rs==='reviewed'||s==='reviewed'&&rs!=='reviewed'||s==='uncertain'&&rs!=='uncertain'||s==='draft'&&rs!=='draft')return false;for(const [el,key]of [['priorityFilter','priority'],['difficultyFilter','difficulty']])if($(el).value!=='all'&&text(r[key])!==$(el).value)return false;if($('labelFilter').value!=='all'&&r.ai_stance!==$('labelFilter').value)return false;if($('batchFilter').value!=='all'&&batch(r)!==$('batchFilter').value)return false;const q=$('searchFilter').value.trim().toLocaleLowerCase(),g=$('groupFilter').value.trim();return(!q||(r.id+' '+r.text+' '+r.source_entity).toLocaleLowerCase().includes(q))&&(!g||group(r).includes(g));}
function render(){filtered=ROWS.filter(matches);if(!filtered.some(r=>r.id===currentId))currentId=filtered[0]?.id||'';const r=BYID.get(currentId);$('empty').style.display=r?'none':'block';$('reviewCard').style.display=r?'block':'none';document.body.classList.toggle('blind',$('blindMode').checked);updateProgress();if(!r)return;const i=filtered.findIndex(x=>x.id===r.id),d=rowState(r);$('position').textContent=(i+1)+' / '+filtered.length;$('previousButton').disabled=i===0;$('nextButton').disabled=i===filtered.length-1;$('target').textContent=r.source_entity;$('rowId').textContent=r.id;renderHeadline(r.text,r.source_entity);$('rowMeta').replaceChildren();for(const v of ['批次 '+batch(r),text(r.priority),text(r.difficulty)])if(v){const e=document.createElement('span');e.className='tag '+v;e.textContent=QUEUE_NAMES[v]||v;$('rowMeta').append(e);}const same=ROWS.filter(x=>group(x)===group(r)&&group(r));$('groupNote').textContent=same.length>1?'同组还有 '+(same.length-1)+' 条目标或相近原文。每个目标分别判断；组号：'+group(r):'本条组号：'+(group(r)||'未提供')+'。目标级标签与新闻整体情绪分开审核。';$('aiLabel').textContent='AI 建议：'+LABELS[r.ai_stance];$('aiRationale').textContent=text(field(r,'ai_rationale','ai_reason_zh','rationale_zh','rationale'));$('aiEvidence').textContent=text(field(r,'evidence_quotes','ai_evidence','evidence'));$('aiUncertainty').textContent=text(field(r,'uncertainty','review_reason','risk_flags'));$('assetSuggestion').textContent='资产草稿：'+(text(field(r,'ai_asset_candidate','proposed_asset_identifier','source_symbol_candidates'))||'无明确候选')+'；映射状态：'+(text(field(r,'mapping_status','source_symbol_match'))||'待核实');$('secondaryAi').style.display=r.secondary_ai_stance?'block':'none';$('secondaryLabel').textContent=r.secondary_ai_stance?(r.ai_disagreement?'两次 AI 建议不同 · ':'二次 AI 建议 · ')+LABELS[r.secondary_ai_stance]:'';$('secondaryReason').textContent=text(r.secondary_ai_reason);$('humanLabel').value=d.human_stance||($('blindMode').checked?'':r.ai_stance);$('humanAsset').value=d.asset_identifier;$('assetVerified').checked=d.asset_verified;$('humanNotes').value=d.notes;$('humanEvent').value=d.event_group_id;$('currentStatus').textContent=d.human_reviewed?'立场已确认 · '+(d.asset_verified?'资产已核实':d.human_stance==='exclude'?'本条排除':'资产待核实')+' · '+d.reviewed_at:status(r.id)==='uncertain'?'待查，尚未批准':decisions[r.id]?'已暂存修改，尚未确认':'尚未人工审核';$('sourceLabel').textContent=$('blindMode').checked?'':'作者原标签：'+text(r.source_sentiment);const provenance={id:r.id,source:field(r,'source','source_url'),source_headline_id:r.source_headline_id,text_group_id:r.text_group_id,group_id:r.group_id,ai_reviewer:r.ai_reviewer,ai_method:r.ai_method,mapping_status:field(r,'mapping_status','source_symbol_match')};$('sourceDetails').textContent=JSON.stringify(provenance,null,2);}
function next(delta){const index=filtered.findIndex(r=>r.id===currentId);currentId=filtered[Math.max(0,Math.min(filtered.length-1,index+delta))]?.id||currentId;render();}
function approve(exclude=false){const r=BYID.get(currentId);if(!r)return;const d={...rowState(r),human_stance:exclude?'exclude':$('humanLabel').value,asset_identifier:$('humanAsset').value.trim(),asset_verified:$('assetVerified').checked,notes:$('humanNotes').value,event_group_id:$('humanEvent').value.trim(),human_reviewed:true,review_status:'reviewed',reviewed_at:new Date().toISOString()};try{const confirmed=ReviewCore.cleanDecision(d,r.id);decisionHistory.push({id:r.id,previous:decisions[r.id]?{...decisions[r.id]}:null});decisions[r.id]=confirmed;}catch(e){toast('请选择有效立场；若勾选资产已核实，还需要填写实际资产标识。',true);return;}persist();const index=filtered.findIndex(x=>x.id===r.id),candidate=filtered[index+1]||filtered[index-1];if(candidate)currentId=candidate.id;toast('立场已确认 '+r.id+(d.asset_verified||d.human_stance==='exclude'?'。':'；资产仍待核实，不是可训练样本。'));render();}
function uncertain(){const r=BYID.get(currentId);if(!r)return;decisionHistory.push({id:r.id,previous:decisions[r.id]?{...decisions[r.id]}:null});decisions[r.id]={...rowState(r),human_stance:rowState(r).human_stance,asset_identifier:$('humanAsset').value.trim(),asset_verified:$('assetVerified').checked,notes:$('humanNotes').value,event_group_id:$('humanEvent').value.trim(),human_reviewed:false,review_status:'uncertain',reviewed_at:''};persist();toast('已标为待查，不计入人工确认。');next(1);}
function merged(r){const d=rowState(r);return {...r,asset_identifier:d.asset_verified?d.asset_identifier:'',asset_identifier_draft:d.asset_verified?'':d.asset_identifier,human_stance:d.human_stance,human_reviewed:!!d.human_reviewed,formal_training_ready:false,asset_verified:!!d.asset_verified,human_notes:d.notes,event_group_id:d.event_group_id,review_status:decisions[r.id]?.review_status||'unreviewed',reviewed_at:d.reviewed_at,dataset_sha256:SHA};}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportJson(approvedOnly){const rows=approvedOnly?ROWS.filter(r=>decisions[r.id]?.human_reviewed):ROWS;download((approvedOnly?'human_reviewed':'all_review_rows')+'_'+SHA.slice(0,10)+'.jsonl',rows.map(r=>JSON.stringify(merged(r))).join('\n')+'\n','application/x-ndjson;charset=utf-8');}
$('workspaceTitle').textContent=DATA.title;$('datasetShort').textContent='数据版本 '+SHA.slice(0,12);document.title=DATA.title;
for(const [id,key]of [['priorityFilter','priority'],['difficultyFilter','difficulty'],['batchFilter','batch_id']]){const values=[...new Set(ROWS.map(r=>key==='batch_id'?batch(r):text(r[key])).filter(Boolean))].sort();for(const value of values){const option=document.createElement('option');option.value=value;option.textContent=key==='batch_id'?'批次 '+value:QUEUE_NAMES[value]||value;$(id).append(option);}}
for(const id of ['statusFilter','priorityFilter','difficultyFilter','labelFilter','batchFilter','disagreementFilter'])$(id).addEventListener('change',render);for(const id of ['searchFilter','groupFilter'])$(id).addEventListener('input',render);
for(const [id,key]of [['humanLabel','human_stance'],['humanAsset','asset_identifier'],['humanNotes','notes'],['humanEvent','event_group_id']])$(id).addEventListener('input',()=>update(key,$(id).value));$('assetVerified').addEventListener('change',()=>update('asset_verified',$('assetVerified').checked));
$('undoButton').onclick=()=>{const last=decisionHistory.pop();if(!last)return;if(last.previous)decisions[last.id]=last.previous;else delete decisions[last.id];currentId=last.id;$('statusFilter').value='all';persist();render();toast('已撤回上一操作，恢复 '+last.id+'。');};$('previousButton').onclick=()=>next(-1);$('nextButton').onclick=()=>next(1);$('approveButton').onclick=()=>approve();$('excludeButton').onclick=()=>approve(true);$('uncertainButton').onclick=uncertain;$('resetButton').onclick=()=>{const d=decisions[currentId];if(d){decisions[currentId]={...d,human_reviewed:false,review_status:'draft',reviewed_at:''};persist();render();toast('本条人工确认已撤回，编辑内容保留。');}};
$('backupButton').onclick=()=>download('review_progress_'+SHA.slice(0,10)+'.json',JSON.stringify(envelope(),null,2),'application/json;charset=utf-8');$('reviewedCsvButton').onclick=()=>download('human_reviewed_'+SHA.slice(0,10)+'.csv',ReviewCore.csv(Object.values(decisions)),'text/csv;charset=utf-8');$('reviewedJsonButton').onclick=()=>exportJson(true);$('allJsonButton').onclick=()=>exportJson(false);$('importButton').onclick=()=>$('importFile').click();
$('importFile').onchange=async()=>{const file=$('importFile').files[0];if(!file)return;try{const imported=ReviewCore.validateImport(JSON.parse(await file.text()),SHA,IDS);for(const [id,d]of Object.entries(imported)){if(decisions[id]&&JSON.stringify(decisions[id])!==JSON.stringify(d))throw Error('本地与导入进度存在冲突：'+id+'。请先备份，再在另一个浏览器副本中恢复，避免覆盖。');}decisions={...decisions,...imported};decisionHistory=[];persist();render();toast('已恢复 '+Object.keys(imported).length+' 条人工操作。');}catch(e){toast('导入失败，现有审核保留：'+e.message,true);}$('importFile').value='';};
document.addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey||e.isComposing||e.target.closest('input,textarea,select,[contenteditable]'))return;const k=e.key.toLowerCase();if(e.repeat&&['a','x','u'].includes(k))return;if(k==='z'){e.preventDefault();$('undoButton').click();}else if(k==='a'){e.preventDefault();approve();}else if(k==='x'){e.preventDefault();approve(true);}else if(k==='u'){e.preventDefault();uncertain();}else if(k==='j'||e.key==='ArrowRight'){e.preventDefault();next(1);}else if(k==='k'||e.key==='ArrowLeft'){e.preventDefault();next(-1);}else if(['1','2','3'].includes(k)){e.preventDefault();const v={1:'bearish',2:'neutral',3:'bullish'}[k];$('humanLabel').value=v;update('human_stance',v);toast('已选择 '+LABELS[v]+'，请确认资产后按 A。');}});
$('blindMode').addEventListener('change',()=>{if($('blindMode').checked)$('labelFilter').value='all';if($('blindMode').checked)$('disagreementFilter').value='all';$('disagreementFilter').disabled=$('blindMode').checked;$('labelFilter').disabled=$('blindMode').checked;render();});$('calibrationMode').addEventListener('change',()=>{if($('calibrationMode').checked){$('blindMode').checked=true;$('labelFilter').value='all';$('labelFilter').disabled=true;$('disagreementFilter').value='all';$('disagreementFilter').disabled=true;$('batchFilter').value='all';$('priorityFilter').value='all';$('difficultyFilter').value='all';$('groupFilter').value='';$('searchFilter').value='';toast('规则校准样本按 AI 类别分层、easy 优先，默认隐藏 AI 建议。它不是最终测试集。');}render();});
render();if(!storeOk)toast('请先导入已有进度，或开始审核并经常导出备份。');
})();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="130U-FinLLM｜资产立场审核工作台")
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.title), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
