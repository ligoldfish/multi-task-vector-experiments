from pathlib import Path
import csv
import hashlib
import json
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'D:\DataSpecificWork\task-vector')
PAYLOAD = ROOT.parent/'modelmate-runbook/payload'
main = ROOT/'原文与代码逐项一致性审计.md'
data = json.loads((ROOT/'requirements-comparison.json').read_text(encoding='utf-8'))
md = main.read_text(encoding='utf-8')
ids = [r['id'] for r in data['requirements']]
assert len(ids)==len(set(ids))==78
assert {i for i in ids if i.startswith('R')} == {f'R{i:02}' for i in range(1,69)}
assert {i for i in ids if i.startswith('I')} == {f'I{i:02}' for i in range(1,11)}
assert len(re.findall(r'^\| R\d{2} /', md, re.M))==68
assert len(re.findall(r'^\| I\d{2} \|', md, re.M))==10
assert len(re.findall(r'^\| B\d{2} \|', md, re.M))==5
assert {f['id'] for f in data['findings']}=={f'A{i:02}' for i in range(1,17)}
assert len(re.findall(r'^### A\d{2} ·',md,re.M))==16
assert sum(data['counts'].values())==68
with (ROOT/'requirements-comparison.csv').open(encoding='utf-8-sig',newline='') as handle:
    csv_rows=list(csv.DictReader(handle))
assert csv_rows==data['requirements']

checked_links=0
for report in [main,ROOT/'data-evaluation-audit.md',ROOT/'model-training-audit.md',ROOT/'loda-decomposition-audit.md']:
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        if not re.match(r'^[A-Z]:[/\\]',target):
            continue
        match=re.match(r'^(.*?)(?::(\d+))?$',target)
        path=Path(match[1])
        assert path.is_file(), (report,path)
        if match[2]:
            assert 1<=int(match[2])<=len(path.read_text(encoding='utf-8-sig').splitlines()),(path,match[2])
        checked_links+=1

prior=json.loads((ROOT/'audit-checks.json').read_text(encoding='utf-8'))
for item in prior['file_hashes']:
    name=item['file']
    assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==item['source_sha256'],name
    assert hashlib.sha256((PAYLOAD/name).read_bytes()).hexdigest()==item['payload_sha256'],name
assert len(prior['file_hashes'])==82
original=Path(r'D:\Download\AppGallery\WeChat Files\wxid_fluchnbzj13i12\FileStorage\File\2026-10\multi_task_vector_experiment_plan.md')
assert original.read_bytes()==(SOURCE/'docs/原始实验方案.md').read_bytes()
assert hashlib.sha256(original.read_bytes()).hexdigest()==data['source_sha256']
result={'validated_utc':datetime.now(timezone.utc).isoformat(),'requirements':68,'inferences':10,'approved_supplements':5,'findings':16,'local_file_links_validated':checked_links,'csv_json_equivalent':True,'source_payload_files_unchanged':82,'original_document_matches':True,'scientific_tests_rerun':False,'model_data_npu_run':False,'main_report_sha256':hashlib.sha256(main.read_bytes()).hexdigest()}
(ROOT/'report-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
