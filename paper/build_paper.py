"""把 paper/abstract.md + paper/body.md 套入大赛 Word 模板，生成 paper/论文.docx（及 PDF 预览）。

流程：组装附录（结果表 + 全部源程序）→ pandoc（以模板为 reference-doc）生成正文 →
把模板的封面/摘要/目录两节注入正文前部，正文节使用模板的页脚（页码从 1 开始、无页眉）。
AI assistance: Claude (Claude Opus 5.5), Anthropic, 2026-10-26. 用于代码实现，详见支撑材料《AI 工具使用详情说明》。
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

P = Path(__file__).resolve().parent
ROOT = P.parent
BUILD = P / 'build'
TITLE = '基于折外校准、风格敏感性分析与潜在高斯最优传输的啤酒风味建模'
KEYWORDS = '感官评分；Extra Trees；置换重要性；风格内置换；经验 CDF；Ledoit–Wolf 收缩；高斯最优传输'
TEAM = 'MCB260XXXX'
TRACK = 'A'


def md_table(df, widths=None):
    cols = list(df.columns)
    widths = widths or [1] * len(cols)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(':' + '-' * (6 * w) for w in widths) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in r.values) + ' |')
    return '\n'.join(lines)


def appendix_tables():
    t = pd.read_csv(ROOT / 'results/q2/q2_v31_target_summary.csv', encoding='utf-8-sig', index_col=0)
    a1 = pd.DataFrame({'感官维度': t.index, 'R²': t.R2_mean_oof.round(3), '支持度': t.support_tier,
                       '解释稳定性': t.interpretation_stability, 'Top-3 候选': t.top3.str.replace(' | ', '；')})
    w = pd.read_csv(ROOT / 'results/q2/q2_within_style_permutation_by_field.csv', encoding='utf-8-sig')
    w = w[w.n_targets >= 3]
    a2 = pd.DataFrame({'化合物': w.field, '出现维度数': w.n_targets, '保留比例中位数': w.median_retention.round(3),
                       '风格内为正比例': (w.share_within_positive * 100).round(1).astype(str) + '%',
                       '风格内仍在Top-10': (w.share_within_top10 * 100).round(1).astype(str) + '%'})
    s = pd.read_csv(ROOT / 'results/q3/sensitivity_envelope_selection.csv', encoding='utf-8-sig')
    s['method'] = s.method.map({'latent_OT': '共同空间OT', 'baseline_optimizer': '首版优化基线'})
    a3 = s.rename(columns={'method': '方法', 'beer': '源样本', 'status_frozen': '冻结判据',
                           'selected_alpha_frozen': '首次达标α', 'status_envelope': '容许域判据',
                           'selected_alpha_envelope': '容许域首次达标α'}).fillna('—')
    return ('表 A1 50 维 Top-3 候选与两维分档\n\n' + md_table(a1, [3, 1, 1, 2, 9]) +
            '\n\n表 A2 风格内置换按化合物汇总（出现于至少 3 个维度的 Top-3）\n\n' + md_table(a2) +
            '\n\n表 A3 迁移逐样本验收（冻结判据与事后容许域判据；容许域 P95 半径 1.183）\n\n' + md_table(a3))


def code_listing():
    files = sorted((ROOT / 'analysis').rglob('*.py')) + sorted((ROOT / 'src').rglob('*.py')) + [P / 'build_paper.py']
    out = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        out.append(f'**{rel}**\n\n```python\n{f.read_text(encoding="utf-8").rstrip()}\n```\n')
    return '\n'.join(out)


def build_markdown():
    body = (P / 'body.md').read_text(encoding='utf-8')
    body = re.sub(r'【附表 A1.*?】', lambda m: appendix_tables(), body, flags=re.S)
    body = body.replace('【全部源程序清单由排版脚本自动附于此处。】', '### 源程序清单\n\n' + code_listing())
    (BUILD / 'paper_full.md').write_text(body, encoding='utf-8')


def pandoc():
    subprocess.run(['pandoc', str(BUILD / 'paper_full.md'), '-o', str(BUILD / 'body.docx'),
                    '--reference-doc', str(P / 'template.docx'), '--resource-path', str(P),
                    '-f', 'markdown+pipe_tables+tex_math_dollars+implicit_figures'], check=True)


# ---------------------------------------------------------------- 样式（模板缺少的正文样式由此补充）
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体" w:cs="Times New Roman"/>'
HEI = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体" w:cs="Times New Roman"/>'
NEW_STYLES = {
    'Heading2': f'<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="1"/><w:next w:val="BodyText"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="200" w:after="80" w:line="300" w:lineRule="auto"/><w:jc w:val="left"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr>{HEI}<w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>',
    'Heading3': f'<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="1"/><w:next w:val="BodyText"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr>{HEI}<w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>',
    'BodyText': f'<w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/><w:basedOn w:val="1"/><w:qFormat/><w:pPr><w:spacing w:before="0" w:after="0" w:line="312" w:lineRule="auto"/><w:ind w:firstLine="480"/><w:jc w:val="both"/></w:pPr><w:rPr>{FONT}<w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>',
    'FirstParagraph': '<w:style w:type="paragraph" w:styleId="FirstParagraph"><w:name w:val="First Paragraph"/><w:basedOn w:val="BodyText"/><w:next w:val="BodyText"/><w:qFormat/></w:style>',
    'Compact': f'<w:style w:type="paragraph" w:styleId="Compact"><w:name w:val="Compact"/><w:basedOn w:val="1"/><w:qFormat/><w:pPr><w:spacing w:before="20" w:after="20"/><w:jc w:val="center"/></w:pPr><w:rPr>{FONT}<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>',
    'ImageCaption': f'<w:style w:type="paragraph" w:styleId="ImageCaption"><w:name w:val="Image Caption"/><w:basedOn w:val="1"/><w:qFormat/><w:pPr><w:spacing w:before="40" w:after="160"/><w:jc w:val="center"/></w:pPr><w:rPr>{HEI}<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>',
    'TableCaption': f'<w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/><w:basedOn w:val="1"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="60"/><w:jc w:val="center"/></w:pPr><w:rPr>{HEI}<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>',
    'CaptionedFigure': '<w:style w:type="paragraph" w:styleId="CaptionedFigure"><w:name w:val="Captioned Figure"/><w:basedOn w:val="1"/><w:pPr><w:keepNext/><w:spacing w:before="120"/><w:jc w:val="center"/></w:pPr></w:style>',
    'Figure': '<w:style w:type="paragraph" w:styleId="Figure"><w:name w:val="Figure"/><w:basedOn w:val="1"/><w:pPr><w:jc w:val="center"/></w:pPr></w:style>',
    'Table': '<w:style w:type="table" w:styleId="Table"><w:name w:val="Table"/><w:basedOn w:val="15"/><w:tblPr><w:jc w:val="center"/><w:tblBorders><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tblBorders><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr><w:tcPr><w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders></w:tcPr></w:tblStylePr></w:style>',
    'SourceCode': '<w:style w:type="paragraph" w:customStyle="1" w:styleId="SourceCode"><w:name w:val="Source Code"/><w:basedOn w:val="1"/><w:link w:val="VerbatimChar"/><w:pPr><w:wordWrap w:val="0"/><w:spacing w:before="0" w:after="0" w:line="220" w:lineRule="exact"/><w:jc w:val="left"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="宋体"/><w:sz w:val="15"/><w:szCs w:val="15"/></w:rPr></w:style>',
}


def fix_styles(styles):
    for sid, xml in NEW_STYLES.items():
        styles = re.sub(rf'<w:style [^>]*w:styleId="{sid}".*?</w:style>', '', styles, flags=re.S)
        styles = styles.replace('</w:styles>', xml + '</w:styles>')
    # 一级标题：居中、三号黑体，压缩段距
    styles = re.sub(r'(<w:style [^>]*w:styleId="2".*?<w:pPr>).*?(</w:pPr><w:rPr>).*?(</w:rPr></w:style>)',
                    lambda m: m.group(1) + '<w:keepNext/><w:keepLines/><w:spacing w:before="360" w:after="200" w:line="300" w:lineRule="auto"/><w:jc w:val="center"/><w:outlineLvl w:val="0"/>'
                    + m.group(2) + HEI + '<w:b/><w:bCs/><w:kern w:val="44"/><w:sz w:val="32"/><w:szCs w:val="32"/>' + m.group(3),
                    styles, count=1, flags=re.S)
    return styles


def fix_body(doc):
    # 表格属性顺序：jc 必须在 tblLook 之前
    doc = re.sub(r'(<w:tblLook [^>]*/>)<w:jc w:val="start" />', r'<w:jc w:val="center"/>\1', doc)
    doc = re.sub(r'(<w:tblLook [^>]*/>)<w:jc w:val="start"/>', r'<w:jc w:val="center"/>\1', doc)
    # “表 X …”段落改用表题样式
    doc = re.sub(r'<w:pStyle w:val="(?:BodyText|FirstParagraph)" ?/>((?:(?!</w:p>).)*?<w:t[^>]*>表 (?:\d|A\d))',
                 r'<w:pStyle w:val="TableCaption"/>\1', doc, flags=re.S)
    return doc


# ---------------------------------------------------------------- docx 拼装
def para_text(pPr, text, rPr=''):
    return f'<w:p>{pPr}<w:r>{rPr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def assemble():
    tpl = zipfile.ZipFile(P / 'template.docx')
    tdoc = tpl.read('word/document.xml').decode('utf-8')
    body_start = tdoc.index('<w:body>') + len('<w:body>')
    secs = [m.end() for m in re.finditer(r'</w:sectPr></w:pPr>.*?</w:p>', tdoc)]
    front = tdoc[body_start:secs[1]]                       # 封面/摘要节 + 目录节
    final_sect = re.search(r'<w:sectPr>(?:(?!<w:sectPr>).)*</w:sectPr></w:body>', tdoc, re.S).group(0)[:-len('</w:body>')]

    # 赛道、题目
    front = re.sub(r'（A/B）', TRACK, front)
    front = front.replace('论文题目', escape(TITLE))
    # 摘要正文：复制占位段落的格式，逐段替换
    m = re.search(r'<w:p\b[^>]*>(?:(?!<w:p\b).)*?（摘要正文）.*?</w:p>', front, re.S)
    ab_p = m.group(0)
    pPr = re.search(r'<w:pPr>.*?</w:pPr>', ab_p, re.S)
    pPr = pPr.group(0) if pPr else ''
    rPr = re.search(r'<w:r>(<w:rPr>.*?</w:rPr>)', ab_p, re.S)
    rPr = rPr.group(1) if rPr else ''
    paras = [x.strip() for x in (P / 'abstract.md').read_text(encoding='utf-8').split('\n\n') if x.strip()]
    if '<w:ind' not in pPr:
        ind = '<w:ind w:firstLineChars="200" w:firstLine="480"/>'
        ind = '<w:spacing w:line="276" w:lineRule="auto"/>' + ind
        pPr = (re.sub(r'(<w:jc |<w:rPr>|</w:pPr>)', ind + r'\1', pPr, count=1)) if pPr else \
            '<w:pPr><w:ind w:firstLineChars="200" w:firstLine="480"/><w:jc w:val="both"/></w:pPr>'
    front = front.replace(ab_p, ''.join(para_text(pPr, x, rPr) for x in paras))
    # 摘要与关键词之间模板留有多个空段落，删去以保证摘要页不溢出
    k = front.index('<w:t>关键词：</w:t>')
    kp = front.rfind('<w:p ', 0, k)
    last_ab = front.index('</w:p>', front.rfind(escape(paras[-1][-12:]), 0, kp))
    seg = front[:kp]
    head, tail = seg[:last_ab], seg[last_ab:]
    tail = re.sub(r'<w:p\b[^>]*>(?:<w:pPr>(?:(?!</w:pPr>).)*</w:pPr>)?</w:p>|<w:p\b[^>]*/>', '', tail, flags=re.S)
    front = head + tail + front[kp:]
    # 关键词
    front = re.sub(r'(<w:t>关键词：</w:t></w:r>)', r'\1<w:r><w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr><w:t xml:space="preserve">'
                   + escape(KEYWORDS) + '</w:t></w:r>', front, count=1)
    # 目录：替换示例条目为 TOC 域
    toc_items = re.findall(r'<w:p\b[^>]*><w:pPr><w:pStyle w:val="(?:10|13|5)"/>.*?</w:p>', front, re.S)
    toc_field = ('<w:p><w:pPr><w:pStyle w:val="10"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9070"/></w:tabs></w:pPr>'
                 '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
                 '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>（目录：在 Word 中右键“更新域”生成）</w:t></w:r>'
                 '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')
    front = front.replace(toc_items[0], toc_field, 1)
    for it in toc_items[1:]:
        front = front.replace(it, '', 1)

    # 读取 pandoc 正文
    src = zipfile.ZipFile(BUILD / 'body.docx')
    doc = fix_body(src.read('word/document.xml').decode('utf-8'))
    rels = src.read('word/_rels/document.xml.rels').decode('utf-8')
    ctypes = src.read('[Content_Types].xml').decode('utf-8')

    # 模板页眉页脚并入，重新编号
    trels = tpl.read('word/_rels/document.xml.rels').decode('utf-8')
    idmap = {}
    for rid, target in re.findall(r'Id="(rId\d+)"[^>]*Target="((?:header|footer)\d\.xml)"', trels):
        new = f'rIdTpl{target.split(".")[0]}'
        idmap[rid] = new
        kind = 'header' if target.startswith('header') else 'footer'
        rels = rels.replace('</Relationships>', f'<Relationship Id="{new}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{kind}" Target="tpl_{target}"/></Relationships>')
        ct = f'application/vnd.openxmlformats-officedocument.wordprocessingml.{kind}+xml'
        ctypes = ctypes.replace('</Types>', f'<Override PartName="/word/tpl_{target}" ContentType="{ct}"/></Types>')
    for old, new in idmap.items():
        front = front.replace(f'r:id="{old}"', f'r:id="{new}"')
        final_sect = final_sect.replace(f'r:id="{old}"', f'r:id="{new}"')

    # 模板的“行网格”会把 1.3 倍行距吸附成两行网格，改为不吸附
    grid = re.compile(r'<w:docGrid [^>]*/>')
    front = grid.sub('<w:docGrid w:linePitch="312"/>', front)
    final_sect = grid.sub('<w:docGrid w:linePitch="312"/>', final_sect)
    b0 = doc.index('<w:body>') + len('<w:body>')
    doc = doc[:b0] + front + doc[b0:]
    doc = re.sub(r'<w:sectPr\b(?:(?!<w:sectPr).)*</w:sectPr>\s*</w:body>', final_sect + '</w:body>', doc, flags=re.S)
    # 模板根元素命名空间（front 中含 w14/v/o 等）
    ns = {}
    for root in (re.search(r'<w:document[^>]*>', doc).group(0), re.search(r'<w:document[^>]*>', tdoc).group(0)):
        for k, v in re.findall(r'(xmlns:\w+)="([^"]*)"', root):
            ns.setdefault(k, v)
    ign = re.search(r'mc:Ignorable="([^"]*)"', re.search(r'<w:document[^>]*>', tdoc).group(0))
    root = '<w:document ' + ' '.join(f'{k}="{v}"' for k, v in ns.items()) + (f' mc:Ignorable="{ign.group(1)}"' if ign else '') + '>'
    doc = re.sub(r'<w:document[^>]*>', root, doc, count=1)

    out = P / '论文.docx'
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == 'word/document.xml':
                data = doc.encode('utf-8')
            elif item.filename == 'word/_rels/document.xml.rels':
                data = rels.encode('utf-8')
            elif item.filename == 'word/styles.xml':
                data = fix_styles(data.decode('utf-8')).encode('utf-8')
            elif item.filename == '[Content_Types].xml':
                data = ctypes.encode('utf-8')
            z.writestr(item, data)
        for target in set(re.findall(r'Target="((?:header|footer)\d\.xml)"', trels)):
            z.writestr(f'word/tpl_{target}', tpl.read(f'word/{target}'))
    return out


if __name__ == '__main__':
    BUILD.mkdir(exist_ok=True)
    build_markdown()
    pandoc()
    out = assemble()
    print(out)
