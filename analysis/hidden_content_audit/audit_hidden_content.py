"""Read-only inspection of original A-track material. Never executes document content.
AI-assisted audit: OpenAI Codex, 2026-10-25.
"""
from pathlib import Path
import csv, hashlib, json, re, unicodedata, zipfile
from collections import Counter
from xml.etree import ElementTree as ET
import fitz
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PACK=next(ROOT.glob('2026年第七届*'))
ADIR=next(PACK.glob('*赛道A*'))
TRIGGERS=re.compile(r'ignore.{0,30}(previous|instructions)|system\s*prompt|prompt\s*injection|'
    r'if\s+you\s+are\s+(an?\s+)?(AI|LLM)|chatgpt|openai|large\s+language\s+model|'
    r'忽略.{0,15}(指令|要求|之前|前面)|如果.{0,15}(人工智能|大模型|AI)|'
    r'(人工智能|大模型|AI).{0,15}(必须|请输出|回答|暗号|检测)|'
    r'隐藏指令|提示词注入|防AI|反AI|防止AI|水印|暗号|彩蛋',re.I)

def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def scan_text(text):
    hits=[{'offset':m.start(),'context':text[max(0,m.start()-60):m.end()+100]} for m in TRIGGERS.finditer(text)]
    weird=Counter(f'U+{ord(ch):04X} {unicodedata.name(ch,"UNKNOWN")}' for ch in text
        if unicodedata.category(ch)=='Cf' or (unicodedata.category(ch)=='Cc' and ch not in '\n\r\t'))
    return {'instruction_keyword_hits':hits[:60], 'special_characters':dict(weird)}

result={'scope':'A-track original PDF/data plus supplied common rules and Word templates; no B-track or historical papers',
        'pdf':[],'xlsx':[],'csv':[],'docx':[],'other_text':[]}
pdfs=list(PACK.glob('*.pdf'))+list(ADIR.glob('*.pdf'))
pdfs += list(ROOT.glob('*.pdf'))
seen=set()
for path in pdfs:
    sha=h(path)
    if sha in seen:continue
    seen.add(sha);doc=fitz.open(path)
    record={'path':str(path),'sha256':sha,'pages':len(doc),'metadata':doc.metadata,
            'embedded_files':doc.embfile_names(),'ocgs':doc.get_ocgs(),
            'xml_metadata_scan':scan_text(doc.get_xml_metadata()),'page_checks':[],
            'metadata_scan':scan_text(json.dumps(doc.metadata,ensure_ascii=False)), 'object_flags':[]}
    for idx,page in enumerate(doc):
        suspicious=[];n=0;covered=[]
        drawings=page.get_drawings()
        for span in page.get_texttrace():
            n+=1
            text=''.join(chr(c[0]) for c in span['chars'] if 0<=c[0]<=0x10ffff)
            if not text.strip():continue
            reasons=[];color=span.get('color',());bbox=fitz.Rect(span['bbox'])
            if span.get('type')==3:reasons.append('invisible_render_mode')
            if span.get('opacity',1)<.05:reasons.append('near_transparent')
            if color and all(v>.94 for v in color):reasons.append('near_white')
            if span.get('size',10)<3:reasons.append('tiny_font_under_3pt')
            if not page.rect.intersects(bbox):reasons.append('outside_page')
            if reasons:suspicious.append({'text':text,'reasons':reasons,'bbox':list(bbox),'color':color,'size':span.get('size')})
            for draw in drawings:
                fill=draw.get('fill')
                if (fill and all(v>.94 for v in fill) and draw.get('fill_opacity',1)>.95
                    and draw.get('seqno',-1)>span.get('seqno',10**9) and draw['rect'].contains(bbox)):
                    covered.append({'text':text,'bbox':list(bbox),'cover_rect':list(draw['rect'])});break
        annotations=[]
        for ann in page.annots() or []:annotations.append({'type':ann.type,'info':ann.info})
        links=page.get_links()
        text=page.get_text()
        record['page_checks'].append({'page':idx+1,'text_spans':n,'suspicious_text_spans':suspicious,
            'later_white_cover_candidates':covered,'annotations':annotations,'links':links,'text_scan':scan_text(text)})
        if path.parent==ADIR:
            page.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(OUT/f'A_page_{idx+1}.png')
            (OUT/f'A_page_{idx+1}_extracted.txt').write_text(text,encoding='utf-8')
    for xr in range(1,doc.xref_length()):
        obj=doc.xref_object(xr)
        flags=[token for token in ['/JavaScript','/OpenAction','/AA','/Launch','/ActualText','/Alt ','/EmbeddedFile','/OCProperties'] if token in obj]
        if flags:record['object_flags'].append({'xref':xr,'flags':flags,'object_excerpt':obj[:1200]})
    data=path.read_bytes();end=data.rfind(b'%%EOF')
    record['bytes_after_final_eof_nonwhitespace']=len(data[end+5:].strip()) if end>=0 else None
    result['pdf'].append(record)

for path in (ADIR/'附件').glob('*.xlsx'):
    book=load_workbook(path,data_only=False,read_only=False)
    rec={'path':str(path),'sha256':h(path),'sheets':[],'defined_names':list(book.defined_names),'special_package_parts':[], 'package_text_findings':[]}
    for sheet in book:
        sr={'name':sheet.title,'state':sheet.sheet_state,
            'hidden_rows':[i for i,v in sheet.row_dimensions.items() if v.hidden],
            'hidden_columns':[i for i,v in sheet.column_dimensions.items() if v.hidden],
            'comments':[],'suspicious_style_cells':[],'text_findings':[],
            'formulas':0,'conditional_formatting_rules':len(sheet.conditional_formatting)}
        for row in sheet:
            for cell in row:
                if cell.value is None:continue
                value=str(cell.value);findings=scan_text(value)
                if findings['instruction_keyword_hits'] or findings['special_characters']:sr['text_findings'].append({'cell':cell.coordinate,**findings})
                if cell.data_type=='f':sr['formulas']+=1
                if cell.comment:sr['comments'].append({'cell':cell.coordinate,'text':cell.comment.text})
                color=cell.font.color;reasons=[]
                if cell.number_format.replace(' ','') in [';;;',';;']:reasons.append('hidden_number_format')
                if color and color.type=='rgb' and str(color.rgb)[-6:].upper()=='FFFFFF':reasons.append('white_font')
                if cell.font.sz is not None and cell.font.sz<3:reasons.append('tiny_font')
                if reasons:sr['suspicious_style_cells'].append({'cell':cell.coordinate,'value':value[:300],'reasons':reasons})
        rec['sheets'].append(sr)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if any(t in name.lower() for t in ['comment','custom','vba','external','embedding','drawing']):rec['special_package_parts'].append(name)
            if name.endswith(('.xml','.rels')):
                raw=z.read(name).decode('utf-8',errors='replace')
                try:visible=' '.join(ET.fromstring(raw).itertext())
                except ET.ParseError:visible=raw
                sf=scan_text(visible)
                if sf['instruction_keyword_hits'] or sf['special_characters']:rec['package_text_findings'].append({'part':name,**sf})
    result['xlsx'].append(rec)

for path in (ADIR/'附件').glob('*.csv'):
    text=path.read_text(encoding='utf-8-sig');rows=list(csv.reader(text.splitlines()))
    result['csv'].append({'path':str(path),'sha256':h(path),'rows':len(rows)-1,'columns':len(rows[0]),
        'row_length_variants':dict(Counter(len(r) for r in rows)),**scan_text(text)})

for path in (ADIR/'附件').glob('*.txt'):
    text=path.read_text(encoding='utf-8-sig')
    result['other_text'].append({'path':str(path),'sha256':h(path),**scan_text(text)})

seen=set();ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
for path in list(PACK.glob('*.docx'))+list(ROOT.glob('*.docx')):
    sha=h(path)
    if sha in seen:continue
    seen.add(sha);rec={'path':str(path),'sha256':sha,'hidden_or_white_runs':[],'package_findings':[],'special_parts':[]}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if any(t in name.lower() for t in ['comment','custom','vba','embedding']):rec['special_parts'].append(name)
            if not name.endswith(('.xml','.rels')):continue
            raw=z.read(name).decode('utf-8',errors='replace')
            try:tree=ET.fromstring(raw)
            except ET.ParseError:continue
            sf=scan_text(' '.join(tree.itertext()))
            if sf['instruction_keyword_hits'] or sf['special_characters']:rec['package_findings'].append({'part':name,**sf})
            for run in tree.findall('.//w:r',ns):
                txt=''.join(run.itertext()).strip()
                if not txt:continue
                hidden=run.find('w:rPr/w:vanish',ns)
                col=run.find('w:rPr/w:color',ns)
                white=col is not None and col.get('{'+ns['w']+'}val','').upper()=='FFFFFF'
                if hidden is not None or white:rec['hidden_or_white_runs'].append({'part':name,'text':txt,'hidden_property':hidden is not None,'white':white})
    result['docx'].append(rec)

(OUT/'audit_details.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
for kind,items in result.items():
    if isinstance(items,list):print(kind,len(items),flush=True)
print('Report data saved',flush=True)
