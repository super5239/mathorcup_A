"""用 LibreOffice（UNO）打开 论文.docx，更新目录与域后导出 PDF 预览。
AI assistance: Claude (Claude Opus 5.5), Anthropic, 2026-10-26. 用于代码实现，详见支撑材料《AI 工具使用详情说明》。
"""
import subprocess
import sys
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def prop(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v
    return p


src = Path(sys.argv[1]).resolve()
pdf = src.with_suffix('.pdf')
proc = subprocess.Popen(['soffice', '--headless', '--invisible', '--norestore',
                         '--accept=socket,host=127.0.0.1,port=2002;urp;'])
ctx = None
for _ in range(60):
    try:
        local = uno.getComponentContext()
        resolver = local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', local)
        ctx = resolver.resolve('uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext')
        break
    except Exception:
        time.sleep(1)
desktop = ctx.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', ctx)
doc = desktop.loadComponentFromURL(src.as_uri(), '_blank', 0, (prop('Hidden', True),))
idx = doc.getDocumentIndexes()
for i in range(idx.getCount()):
    idx.getByIndex(i).update()
doc.getTextFields().refresh()
for i in range(idx.getCount()):
    idx.getByIndex(i).update()
doc.storeToURL(pdf.as_uri(), (prop('FilterName', 'writer_pdf_Export'),))
doc.close(True)
try:
    desktop.terminate()
except Exception:
    pass
proc.wait(timeout=30)
print(pdf)
